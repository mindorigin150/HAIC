from __future__ import annotations

import ast
import sys
import tempfile
import types
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from active_adaptation.vla import backend as _backend

from scripts import haic_vla
from active_adaptation.learning.ppo.haic_actor import (
    HaicStudentActor,
    _ACTOR_ADAPT_KEYS,
    extract_actor_adapt_state_dict,
    load_actor_adapt,
)
from active_adaptation.vla.runtime import (
    student_actor_from_policy,
    teacher_latent,
    vla_observations,
)
from active_adaptation.vla import runtime


class _FakeTensorDict(dict):
    def __getitem__(self, key):
        if isinstance(key, tuple):
            value = self
            for part in key:
                value = dict.__getitem__(value, part)
            return value
        return super().__getitem__(key)

    def clone(self, _):
        return _FakeTensorDict(self)


class _FakeEvalEnv:
    device = torch.device("cpu")

    def __init__(self, done_steps):
        self.done_steps = done_steps
        self.num_envs = len(done_steps)
        self.step_count = 0
        self.events = []
        reference_positions = torch.zeros(3, 1, 3)
        reference_positions[:, 0, 0] = torch.tensor([0.0, 0.5, 1.0])
        command_manager = SimpleNamespace(
            t=torch.zeros(self.num_envs, dtype=torch.long),
            motion_len=torch.full((self.num_envs,), 3, dtype=torch.long),
            motion_starts=torch.zeros(self.num_envs, dtype=torch.long),
            motion_ends=torch.full((self.num_envs,), 3, dtype=torch.long),
            object_body_id_motion=0,
            dataset=SimpleNamespace(
                data=SimpleNamespace(body_pos_w=reference_positions)
            ),
            object=SimpleNamespace(
                data=SimpleNamespace(
                    root_link_pos_w=torch.zeros(self.num_envs, 3)
                )
            ),
        )
        self.base_env = SimpleNamespace(
            eval=lambda: self.events.append("eval"),
            command_manager=command_manager,
        )

    def reset(self):
        self.events.append("reset")
        return _FakeTensorDict()

    def step_and_maybe_reset(self, _action):
        self.step_count += 1
        self.base_env.command_manager.t.fill_(self.step_count)
        self.base_env.command_manager.object.data.root_link_pos_w[:, 0].fill_(
            self.step_count / 2
        )
        done = torch.tensor(
            [self.step_count >= done_step for done_step in self.done_steps]
        ).unsqueeze(-1)
        return (
            _FakeTensorDict(
                {"next": _FakeTensorDict({"done": done, "stats": {"success": done}})}
            ),
            self.reset(),
        )


class _FakePool:
    def close(self):
        pass


class _FakeActor:
    def __call__(self, actor_input):
        return torch.zeros(actor_input.shape[0], 23)


class _EncodedTensorDict(dict):
    def clone(self, _):
        return _EncodedTensorDict(self)


def _native_env_method(name):
    # These tensor-only methods can be checked without starting the Isaac SDK.
    path = Path(__file__).resolve().parents[1] / "active_adaptation/envs/base.py"
    tree = ast.parse(path.read_text())
    env_class = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "_Env")
    method = next(node for node in env_class.body if isinstance(node, ast.FunctionDef) and node.name == name)
    namespace = {"torch": torch}
    exec(compile(ast.Module(body=[method], type_ignores=[]), str(path), "exec"), namespace)
    return namespace[name]


class HaicVlaContractTest(unittest.TestCase):
    def setUp(self):
        # Backend fixtures use dicts, not transformed TensorDicts.
        step_mdp = patch.object(_backend, "step_mdp", lambda td: td["next"])
        step_mdp.start()
        self.addCleanup(step_mdp.stop)

    def test_seeded_uniform_preserves_feature_and_per_environment_bounds(self):
        draw = _native_env_method("random_uniform")
        low = torch.tensor([10.0, 20.0])
        env = SimpleNamespace(device="cpu", _episode_generators=[
            torch.Generator().manual_seed(1), torch.Generator().manual_seed(2),
        ])
        samples = draw(env, low, low + 1.0, (2, 2), env_ids=torch.arange(2))
        self.assertTrue(torch.all((samples >= low) & (samples <= low + 1.0)))
        per_env_low = torch.tensor([[30.0, 40.0], [50.0, 60.0]])
        samples = draw(env, per_env_low, per_env_low + 1.0, (2, 2), env_ids=torch.arange(2))
        self.assertTrue(torch.all((samples >= per_env_low) & (samples <= per_env_low + 1.0)))

    def test_rgb_frames_compacts_noncontiguous_slots(self):
        output = torch.arange(8 * 1 * 1 * 4, dtype=torch.uint8).reshape(8, 1, 1, 4)
        env = SimpleNamespace(
            base_env=SimpleNamespace(
                scene={
                    "vla_camera": SimpleNamespace(
                        data=SimpleNamespace(output={"rgb": output})
                    )
                }
            )
        )

        frames = runtime.rgb_frames(env, [2, 7])

        self.assertEqual(frames.shape, (2, 1, 1, 3))
        np.testing.assert_array_equal(frames[0], output[2, ..., :3].numpy())
        np.testing.assert_array_equal(frames[1], output[7, ..., :3].numpy())

    def test_vla_observations_keep_global_ids_for_compact_slots(self):
        rgb = np.asarray(
            [
                [[[1, 2, 3]]],
                [[[4, 5, 6]]],
            ],
            dtype=np.uint8,
        )
        state = np.asarray([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)

        observations = vla_observations(rgb, state, [2, 7], [101, 102], 11)

        self.assertEqual([item.metadata["slot_id"] for item in observations], [2, 7])
        self.assertEqual(
            [item.metadata["action_noise_seed"] for item in observations], [101, 102]
        )
        np.testing.assert_array_equal(
            observations[0].metadata["env_raw_rgb_frame_stack"], rgb[0][None]
        )
        np.testing.assert_array_equal(
            observations[1].metadata["haic_state"], state[1]
        )
        self.assertIsNone(observations[0].data)

    def test_predict_vla_batches_compact_observations_with_their_slots(self):
        seen = []

        def predict_batch(observations):
            seen.extend(
                (
                    item.metadata["slot_id"],
                    item.metadata["haic_state"][0],
                    item.metadata["env_raw_rgb_frame_stack"][0, 0, 0, 0],
                )
                for item in observations
            )
            return [
                SimpleNamespace(
                    action_chunk=np.repeat(
                        item.metadata["haic_state"][None], 40, axis=0
                    ),
                )
                for item in observations
            ]

        output = runtime.predict_vla(
            SimpleNamespace(predict_batch=predict_batch),
            np.arange(5, dtype=np.uint8).reshape(5, 1, 1, 1),
            np.arange(5, dtype=np.float32).reshape(5, 1),
            [2, 7, 11, 19, 23],
            [100, 101, 102, 103, 104],
            step=0,
            batch_size=2,
        )

        np.testing.assert_array_equal(
            output[:, :, 0], np.repeat(np.arange(5)[:, None], 40, axis=1)
        )
        self.assertEqual(
            seen,
            [(2, 0, 0), (7, 1, 1), (11, 2, 2), (19, 3, 3), (23, 4, 4)],
        )

    def test_teacher_latent_is_only_the_consumed_privileged_feature(self):
        policy = SimpleNamespace(
            object_transform=lambda encoded: None,
            encoder_priv=lambda encoded: encoded.__setitem__(
                "priv_feature", torch.zeros(2, 256)
            ),
        )
        tensordict = _EncodedTensorDict()
        self.assertEqual(teacher_latent(policy, tensordict).shape, (2, 256))

    def test_teacher_latent_rollout_emits_the_scalar_action(self):
        policy = SimpleNamespace(
            object_transform=lambda encoded: None,
            encoder_priv=lambda encoded: encoded.__setitem__(
                "priv_feature", torch.ones(2, 256)
            ),
        )
        tensordict = _EncodedTensorDict({"policy": torch.zeros(2, 1)})
        output = runtime.teacher_latent_rollout(policy, tensordict)
        self.assertEqual(output["action"].shape, (2, 256))

    def test_student_actor_tensor_contract(self):
        actor = HaicStudentActor()
        actor_input = torch.zeros(2, 861)
        loc = actor(actor_input)
        self.assertEqual(loc.shape, (2, 23))

    def test_vla_runtime_actor_is_a_fixed_native_copy(self):
        actor = HaicStudentActor()
        native_state = {
            source_key: actor.state_dict()[target_key]
            for target_key, source_key in _ACTOR_ADAPT_KEYS.items()
        }
        policy = SimpleNamespace(
            actor_adapt=SimpleNamespace(state_dict=lambda: native_state)
        )

        fixed_actor = student_actor_from_policy(policy, torch.device("cpu"))

        self.assertFalse(
            any(parameter.requires_grad for parameter in fixed_actor.parameters())
        )

    def test_native_actor_adapt_key_extraction(self):
        actor = HaicStudentActor()
        native_state = {
            source_key: actor.state_dict()[target_key]
            for target_key, source_key in _ACTOR_ADAPT_KEYS.items()
        }
        actor.load_state_dict(extract_actor_adapt_state_dict(native_state))

    def test_native_checkpoint_exports_an_isaac_free_frozen_actor(self):
        actor = HaicStudentActor()
        native_state = {
            source_key: actor.state_dict()[target_key]
            for target_key, source_key in _ACTOR_ADAPT_KEYS.items()
        }
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory) / "native.pt"
            torch.save({"policy": {"actor_adapt": native_state}}, checkpoint)
            exported = load_actor_adapt(checkpoint)

        self.assertFalse(any(parameter.requires_grad for parameter in exported.parameters()))
        for name, parameter in actor.state_dict().items():
            torch.testing.assert_close(parameter, exported.state_dict()[name])

    def test_vector_backend_maps_latent_through_fixed_actor(self):
        from active_adaptation.vla.backend import HaicEnvStepBackend
        from latency_bench.core.types import Action

        seen = []
        actor_inputs = []
        next_td = _FakeTensorDict(
            {
                "done": torch.tensor([[True]]),
                "truncated": torch.tensor([[False]]),
                "reward": torch.tensor([[2.0]]),
                "stats": {"success": torch.tensor([[1]])},
            }
        )
        carry = _FakeTensorDict(
            {
                "command": torch.zeros(1, 0),
                "policy": torch.zeros(1, 605),
            }
        )
        command = types.SimpleNamespace(
            t=torch.zeros(1, dtype=torch.long),
            motion_len=torch.full((1,), 3, dtype=torch.long),
            object=types.SimpleNamespace(
                data=types.SimpleNamespace(root_link_pos_w=torch.zeros(1, 3))
            ),
        )
        active_ids = torch.arange(1)
        base_env_obj = types.SimpleNamespace(
            device=torch.device("cpu"),
            step_dt=0.02,
            action_dim=23,
            command_manager=command,
            active_env_ids=active_ids,
            set_active_env_ids=lambda slot_ids: None,
        )

        class Env:
            num_envs = 1
            base_env = base_env_obj

            def step(self, action):
                seen.append(action["action"])
                return _FakeTensorDict({"next": next_td})

            def close(self):
                pass

        torchrl = types.ModuleType("torchrl")
        torchrl_envs = types.ModuleType("torchrl.envs")
        torchrl_utils = types.ModuleType("torchrl.envs.utils")
        torchrl_utils.step_mdp = lambda td: td["next"]
        with patch.dict(
            sys.modules,
            {
                "torchrl": torchrl,
                "torchrl.envs": torchrl_envs,
                "torchrl.envs.utils": torchrl_utils,
            },
        ):
            backend = HaicEnvStepBackend(
                Env(),
                lambda actor_input: (
                    actor_inputs.append(actor_input.clone())
                    or torch.zeros(1, 23)
                ),
            )
            backend._carry = carry
            backend._motion_len[0] = torch.tensor(3)
            backend._cart_start[0] = torch.zeros(3)
            backend._ref_cart_displacement[0] = torch.ones(3)
            result = backend.step_slots(
                {0: Action(value=np.ones(256, dtype=np.float32))}
            )

        self.assertTrue(result[0].result.done)
        self.assertEqual(result[0].result.reward, 2.0)
        self.assertEqual(result[0].result.info["task_metrics"]["success"], 1)
        np.testing.assert_array_equal(actor_inputs[0][0, 605:].numpy(), np.ones(256))
        self.assertEqual(seen[0].shape, (1, 23))

    def test_vector_backend_advances_only_tail_active_slots(self):
        from active_adaptation.vla.backend import HaicEnvStepBackend
        from latency_bench.core.types import Action

        active_ids = torch.arange(2)
        command = SimpleNamespace(
            t=torch.tensor([5, 1]),
            motion_len=torch.tensor([6, 4]),
            object=SimpleNamespace(
                data=SimpleNamespace(root_link_pos_w=torch.zeros(2, 3))
            ),
        )
        base_env_obj = SimpleNamespace(
            device=torch.device("cpu"),
            step_dt=0.02,
            action_dim=23,
            command_manager=command,
        )

        def set_active_env_ids(slot_ids):
            nonlocal active_ids
            active_ids = torch.as_tensor(slot_ids)

        base_env_obj.active_env_ids = active_ids
        base_env_obj.set_active_env_ids = set_active_env_ids
        actor_inputs = []
        motor_actions = []

        class Env:
            num_envs = 2
            base_env = base_env_obj

            def step(self, action):
                motor_actions.append(action["action"].clone())
                command.t[active_ids] += 1
                return _FakeTensorDict(
                    {
                        "next": _FakeTensorDict(
                            {
                                "done": torch.tensor([[False], [False]]),
                                "truncated": torch.tensor([[False], [False]]),
                                "reward": torch.zeros(2, 1),
                                "stats": {"success": torch.zeros(2, 1)},
                            }
                        )
                    }
                )

            def close(self):
                pass

        backend = HaicEnvStepBackend(
            Env(),
            lambda actor_input: (
                actor_inputs.append(actor_input.clone())
                or torch.full((len(actor_input), 23), 7.0)
            ),
        )
        backend._carry = _FakeTensorDict(
            {"command": torch.zeros(2, 0), "policy": torch.zeros(2, 605)}
        )
        backend._motion_len = [torch.tensor(6), torch.tensor(4)]
        backend._cart_start = [torch.zeros(3), torch.zeros(3)]
        backend._ref_cart_displacement = [torch.ones(3), torch.ones(3)]

        responses = backend.step_slots({1: Action(value=np.ones(256, dtype=np.float32))})

        self.assertEqual(list(responses), [1])
        self.assertEqual(actor_inputs[0].shape, (1, 861))
        torch.testing.assert_close(motor_actions[0][0], torch.zeros(23))
        torch.testing.assert_close(motor_actions[0][1], torch.full((23,), 7.0))
        torch.testing.assert_close(command.t, torch.tensor([5, 2]))

    def test_reset_slot_does_not_consume_another_slots_rng(self):
        from active_adaptation.vla.backend import HaicEnvStepBackend

        active_ids = torch.arange(2)
        generators = [torch.Generator().manual_seed(10), torch.Generator().manual_seed(20)]
        command = SimpleNamespace(
            motion_len=torch.ones(2, dtype=torch.long),
            motion_starts=torch.zeros(2, dtype=torch.long),
            motion_ends=torch.ones(2, dtype=torch.long),
            object_body_id_motion=0,
            dataset=SimpleNamespace(data=SimpleNamespace(body_pos_w=torch.zeros(1, 1, 3))),
            object=SimpleNamespace(data=SimpleNamespace(root_link_pos_w=torch.zeros(2, 3))),
        )
        base_env_obj = SimpleNamespace(
            device=torch.device("cpu"),
            step_dt=0.02,
            action_dim=23,
            command_manager=command,
        )

        def set_active_env_ids(slot_ids):
            nonlocal active_ids
            active_ids = torch.as_tensor(slot_ids)

        def set_episode_seed(slot_id, seed):
            generators[slot_id] = torch.Generator().manual_seed(seed)

        base_env_obj.active_env_ids = active_ids
        base_env_obj.set_active_env_ids = set_active_env_ids
        base_env_obj.set_episode_seed = set_episode_seed

        class Env:
            num_envs = 2
            base_env = base_env_obj

            def reset(self, _reset_td):
                torch.rand(1, generator=generators[active_ids.item()])
                return _FakeTensorDict({"command": torch.zeros(2, 0), "policy": torch.zeros(2, 605)})

            def close(self):
                pass

        backend = HaicEnvStepBackend(Env(), lambda actor_input: torch.zeros(len(actor_input), 23))
        backend._carry = _FakeTensorDict({"command": torch.zeros(2, 0), "policy": torch.zeros(2, 605)})
        before = generators[1].get_state()

        backend._reset_slot_state(0, seed=123)

        self.assertTrue(torch.equal(generators[1].get_state(), before))

    def test_latency_eval_leaves_policy_in_common_executor(self):
        config = {
            "env": {"obs_fps": 50},
            "executor": {"inference_devices": ["cuda:0"]},
            "logging": {"output_dir": "output"},
        }
        backend = object()
        with patch(
            "active_adaptation.vla.backend.build_haic_env_backend",
            return_value=backend,
        ) as build_backend, patch(
            "latency_bench.eval.driver.run_from_config"
        ) as run_from_config:
            self.assertEqual(
                haic_vla._run_latency_eval(config, object(), object()),
                {"output_dir": "output"},
            )

        build_backend.assert_called_once()
        self.assertEqual(build_backend.call_args.kwargs['obs_fps'], 50)
        run_from_config.assert_called_once_with(
            config,
            env_backend=backend,
            policy=None,
            inference_devices=["cuda:0"],
        )

    def test_dagger_flush_preserves_per_step_action_distillation(self):
        row = {
            "rgb": np.zeros((2, 2, 3), dtype=np.uint8),
            "state": np.zeros(605, dtype=np.float32),
            "action": np.zeros(256, dtype=np.float32),
            "actor_input": np.zeros(605, dtype=np.float32),
            "teacher_action": np.zeros(23, dtype=np.float32),
            "termination": np.asarray(False),
        }
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            with patch.object(
                haic_vla,
                "_encode_video",
                side_effect=lambda _frames, path: path.write_bytes(b"video"),
            ):
                haic_vla._flush_dagger(output, 0, [row])
            with np.load(
                output / "rollout_shards/train/episode_000000/episode.npz"
            ) as payload:
                self.assertEqual(payload["actor_input"].shape, (1, 605))
                self.assertEqual(payload["teacher_action"].shape, (1, 23))
                self.assertEqual(payload["termination"].shape, (1,))
                self.assertTrue(payload["termination"][0])

    def test_dagger_collect_refreshes_each_control_step_at_50hz(self):
        class Env:
            num_envs = 1
            device = torch.device("cpu")

            def reset(self):
                return _FakeTensorDict()

            def step_and_maybe_reset(self, _action):
                return (
                    _FakeTensorDict(
                        {"next": _FakeTensorDict({"done": torch.ones(1, 1, dtype=torch.bool)})}
                    ),
                    self.reset(),
                )

        args = SimpleNamespace(
            output_dir=Path(),
            mode="dagger-collect",
            seed=0,
            row_budget=1,
            dagger_round=0,
            vla_cadence=2,
            action_horizon=40,
            inference_batch_size=1,
        )
        refresh_calls = []

        def refresh_rgb(_env, slots=None, *, update_hz=10):
            refresh_calls.append((slots, update_hz))
            count = 1 if slots is None else len(slots)
            return np.zeros((count, 1, 1, 3), dtype=np.uint8)

        def predict_vla(_pool, _rgb, _state, slots, _episode_seeds, _step, _batch_size):
            return np.zeros((len(slots), 40, 256), dtype=np.float32)

        with patch.object(haic_vla, "_new_pool", return_value=_FakePool()), patch.object(
            haic_vla, "_flush_dagger", return_value=0.0
        ), patch.object(
            runtime, "canonical_state", return_value=torch.zeros(1, 605)
        ), patch.object(
            runtime, "refresh_rgb", side_effect=refresh_rgb
        ), patch.object(
            runtime, "predict_vla", side_effect=predict_vla
        ), patch.object(
            runtime, "teacher_latent", return_value=torch.zeros(1, 256)
        ), patch.object(
            runtime, "teacher_action", return_value=torch.zeros(1, 23)
        ), patch.object(
            runtime, "student_actor_from_policy", return_value=_FakeActor()
        ):
            with tempfile.TemporaryDirectory() as output_dir:
                args.output_dir = Path(output_dir)
                result = haic_vla._dagger_collect(
                    args,
                    Env(),
                    SimpleNamespace(),
                    SimpleNamespace(is_running=lambda: True),
                )

        self.assertEqual(refresh_calls, [(None, 50)])
        self.assertEqual(result["rows"], 1)

    def test_oracle_eval_uses_the_fixed_actor_without_a_vla_pool(self):
        env = _FakeEvalEnv([1, 2])
        args = SimpleNamespace(
            max_steps=3,
            output_dir=Path(),
            mode="oracle-eval",
            vla_cadence=1,
        )
        with patch.object(
            runtime, "canonical_state", return_value=torch.zeros(2, 2)
        ), patch.object(
            runtime,
            "teacher_latent",
            return_value=torch.zeros(2, 256),
        ), patch.object(
            runtime, "student_actor_from_policy", return_value=_FakeActor()
        ):
            with tempfile.TemporaryDirectory() as output_dir:
                args.output_dir = Path(output_dir)
                result = haic_vla._oracle_eval(
                    args, env, SimpleNamespace(), SimpleNamespace()
                )
                self.assertEqual(result["episodes"], 2)
                self.assertEqual(result["successes"], 2)
                self.assertEqual(result["success_rate"], 1.0)
                self.assertTrue((args.output_dir / "oracle-eval.json").exists())

    def test_gae_applies_the_per_transition_discount(self):
        from active_adaptation.learning.ppo.common import GAE

        gae = GAE(gamma=0.9, lmbda=1.0)
        reward = torch.tensor([[[1.0], [2.0]]])
        terminated = torch.zeros_like(reward, dtype=torch.bool)
        done = torch.zeros_like(reward, dtype=torch.bool)
        value = torch.zeros_like(reward)
        next_value = torch.tensor([[[3.0], [4.0]]])
        discount = torch.tensor([[[0.5], [0.25]]])

        advantage, returns = gae(
            reward, terminated, done, value, next_value, discount
        )

        torch.testing.assert_close(
            advantage, torch.tensor([[[3.655], [2.9]]])
        )
        torch.testing.assert_close(returns, advantage)

    def test_latency_decoder_receives_normalized_intermediate_carries(self):
        from tensordict import TensorDict

        class Latency:
            last_submission = [{"latency_ms": 0.0}]
            last_application = [None]
            last_dropped = [False]

            def submit(self, commands, env_ids):
                assert commands.shape == (1, 1)
                return torch.ones(1, dtype=torch.bool)

            def actions(self, env_ids):
                return torch.zeros(1, 1)

            def advance(self, env_ids):
                pass

            def reset(self, env_ids):
                pass

        env = SimpleNamespace(num_envs=1)
        env.cfg = types.SimpleNamespace(
            latency_command_horizon=1,
            latent_dim=1,
            latency_control_repeat=2,
            latency_gamma=0.9,
        )
        env.active_env_ids = torch.tensor([0])
        env.device = torch.device("cpu")
        env.discount = torch.ones(1, 1)
        env.command_latency = Latency()
        env.latency_decoder_inputs = []
        env.latency_decoder = lambda _command, carry: (
            env.latency_decoder_inputs.append(carry["policy"].clone())
            or torch.zeros(1, 1)
        )
        env.set_active_env_ids = lambda ids: setattr(
            env, "active_env_ids", torch.as_tensor(ids)
        )
        env._step_raw = lambda _td: TensorDict(
            {
                "command": torch.zeros(1, 1),
                "policy": torch.tensor([[14.0]]),
                "reward": torch.ones(1, 1),
                "done": torch.zeros(1, 1, dtype=torch.bool),
            },
            batch_size=[1],
        )
        env.latency_observation_norm = lambda td: td.set(
            "policy", (td["policy"] - 10.0) / 2.0
        )

        _native_env_method("_step_latency")(
            env,
            TensorDict(
                {"action": torch.zeros(1, 1), "policy": torch.tensor([[1.0]])},
                batch_size=[1],
            )
        )

        torch.testing.assert_close(
            torch.cat(env.latency_decoder_inputs), torch.tensor([[1.0], [2.0]])
        )

    def test_h1_latency_uses_scalar_commands(self):
        from tensordict import TensorDict

        submitted = []

        class Latency:
            last_submission = [None, None]
            last_application = [None, None]
            last_dropped = [False, False]

            def submit(self, commands, env_ids):
                submitted.append(commands.shape)
                self.last_submission = [{"shape": tuple(commands.shape), "latency_ms": 0.0}] * 2
                return torch.ones(2, dtype=torch.bool)

            def actions(self, env_ids):
                return torch.zeros(2, 3)

            def advance(self, env_ids):
                pass

            def reset(self, env_ids):
                pass

        env = SimpleNamespace(
            num_envs=2,
            cfg=types.SimpleNamespace(
                latency_command_horizon=1,
                latent_dim=3,
                latency_control_repeat=1,
                latency_gamma=0.9,
            ),
            active_env_ids=torch.tensor([0, 1]),
            device=torch.device("cpu"),
            discount=torch.ones(2, 1),
            command_latency=Latency(),
            latency_decoder=lambda command, carry: torch.zeros(2, 1),
            latency_observation_norm=lambda td: td,
            set_active_env_ids=lambda ids: setattr(env, "active_env_ids", torch.as_tensor(ids)),
            _step_raw=lambda td: TensorDict(
                {"reward": torch.ones(2, 1), "done": torch.zeros(2, 1, dtype=torch.bool)},
                batch_size=[2],
            ),
        )
        _native_env_method("_step_latency")(
            env,
            TensorDict({"action": torch.zeros(2, 3)}, batch_size=[2]),
        )
        self.assertEqual(submitted, [(2, 3)])

    def test_motion_resampling_preserves_static_geometry_and_aligns_contacts(self):
        import importlib

        # Motion interpolation does not use IsaacLab's scene-name resolver.
        with patch.dict(
            sys.modules,
            {
                "isaaclab.utils.string": SimpleNamespace(resolve_matching_names=None),
            },
        ):
            motion = importlib.import_module("active_adaptation.utils.motion")

        values = np.arange(6, dtype=np.float32)[:, None]
        np.testing.assert_array_equal(
            motion.nearest_frame_sample(values, source_fps=50, target_fps=20)[:, 0],
            [0.0, 2.0, 5.0],
        )

        object_points = np.arange(3, dtype=np.float32)[None, :, None]
        resampled = motion.interpolate(
            {
                "body_pos_w": np.zeros((6, 1, 3), dtype=np.float32),
                "body_lin_vel_w": np.zeros((6, 1, 3), dtype=np.float32),
                "body_quat_w": np.tile(
                    np.array([1, 0, 0, 0], dtype=np.float32), (6, 1, 1)
                ),
                "body_ang_vel_w": np.zeros((6, 1, 3), dtype=np.float32),
                "joint_pos": np.arange(6, dtype=np.float32)[:, None],
                "joint_vel": np.zeros((6, 1), dtype=np.float32),
                "object_points": object_points,
            },
            source_fps=50,
            target_fps=20,
        )
        np.testing.assert_allclose(resampled["joint_pos"][:, 0], [0.0, 2.5, 5.0])
        self.assertEqual(resampled["joint_pos"].dtype, np.float32)
        self.assertEqual(resampled["body_quat_w"].dtype, np.float32)
        np.testing.assert_array_equal(resampled["object_points"], object_points)

if __name__ == "__main__":
    unittest.main()
