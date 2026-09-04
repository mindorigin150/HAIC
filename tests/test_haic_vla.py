from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

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


class HaicVlaContractTest(unittest.TestCase):
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

        observations = vla_observations(rgb, state, [2, 7], 11)

        self.assertEqual([item.metadata["slot_id"] for item in observations], [2, 7])
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

    def test_profile_env_maps_vla_latent_through_fixed_actor(self):
        seen = []

        class Env:
            device = torch.device("cpu")

            def __init__(self, success):
                self.success = success

            def step_and_maybe_reset(self, action):
                seen.append(action["action"])
                return (
                    _FakeTensorDict(
                        {
                            "next": _FakeTensorDict(
                                {
                                    "done": torch.tensor([[True]]),
                                    "truncated": torch.tensor([[False]]),
                                    "reward": torch.tensor([[2.0]]),
                                    "stats": {
                                        "success": torch.tensor([[self.success]])
                                    },
                                }
                            )
                        }
                    ),
                    self.carry,
                )

        class Actor:
            def __call__(self, actor_input):
                seen.append(actor_input)
                return torch.zeros(1, 23)

        env = Env(True)
        env.carry = _FakeTensorDict(
            {
                "command": torch.zeros(1, 0),
                "policy": torch.zeros(1, 605),
            }
        )
        adapter = haic_vla._HaicProfileEnv(env, Actor())
        adapter._carry = env.carry
        self.assertEqual(seen, [])
        result = adapter.step(
            haic_vla.Action(value=np.ones(256, dtype=np.float32))
        )

        self.assertTrue(result.done)
        self.assertEqual(result.reward, 2.0)
        self.assertEqual(result.info, {"task_metrics": {"success": 1}})
        self.assertEqual(seen[0].shape, (1, 861))
        np.testing.assert_array_equal(seen[0][0, 605:].numpy(), np.ones(256))
        self.assertEqual(seen[1].shape, (1, 23))

        env = Env(False)
        env.carry = _FakeTensorDict(
            {
                "command": torch.zeros(1, 0),
                "policy": torch.zeros(1, 605),
            }
        )
        adapter = haic_vla._HaicProfileEnv(env, Actor())
        adapter._carry = env.carry
        result = adapter.step(
            haic_vla.Action(value=np.ones(256, dtype=np.float32))
        )
        self.assertEqual(result.info, {"task_metrics": {"success": 0}})

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
            row_budget=1,
            dagger_round=0,
            vla_cadence=2,
            inference_batch_size=1,
        )
        refresh_calls = []

        def refresh_rgb(_env, slots=None, *, update_hz=10):
            refresh_calls.append((slots, update_hz))
            count = 1 if slots is None else len(slots)
            return np.zeros((count, 1, 1, 3), dtype=np.uint8)

        def predict_vla(_pool, _rgb, _state, slots, _step, _batch_size):
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

    def test_dagger_eval_counts_only_first_done_per_env(self):
        env = _FakeEvalEnv([1, 2])
        prediction_slots = []

        def predict_vla(_pool, _rgb, _state, slots, _step, _batch_size):
            prediction_slots.append(tuple(slots))
            return np.zeros((len(slots), 40, 256), dtype=np.float32)

        args = SimpleNamespace(
            max_steps=3,
            output_dir=Path(),
            mode="dagger-eval",
            vla_cadence=1,
            inference_batch_size=2,
        )
        with patch.object(haic_vla, "_new_pool", return_value=_FakePool()), patch.object(
            runtime, "canonical_state", return_value=torch.zeros(2, 2)
        ), patch.object(
            runtime,
            "refresh_rgb",
            return_value=np.zeros((2, 1, 1, 3), dtype=np.uint8),
        ), patch.object(runtime, "predict_vla", side_effect=predict_vla), patch.object(
            runtime, "student_actor_from_policy", return_value=_FakeActor()
        ):
            with tempfile.TemporaryDirectory() as output_dir:
                args.output_dir = Path(output_dir)
                result = haic_vla._dagger_eval(
                    args, env, SimpleNamespace(), SimpleNamespace()
                )
                self.assertEqual(result["episodes"], 2)
                self.assertEqual(result["successes"], 2)
                self.assertEqual(result["success_rate"], 1.0)
                self.assertEqual(
                    result["motion_progress"], {"mean": 0.25, "std": 0.25}
                )
                self.assertEqual(
                    result["cart_progress"], {"mean": 0.25, "std": 0.25}
                )
                self.assertEqual(
                    result["pullcart_score"], {"mean": 25.0, "std": 25.0}
                )
                self.assertNotIn("steps", result)
                self.assertEqual(prediction_slots, [(0, 1), (1,)])
                self.assertEqual(env.events[:2], ["eval", "reset"])

    def test_dagger_eval_consumes_distinct_action_chunk_steps(self):
        env = _FakeEvalEnv([3])
        consumed = []

        class RecordingActor:
            def __call__(self, actor_input):
                consumed.append(actor_input[:, 605:].clone())
                return torch.zeros(actor_input.shape[0], 23)

        def predict_vla(_pool, _rgb, _state, slots, _step, _batch_size):
            chunk = np.arange(40, dtype=np.float32)[:, None]
            return np.repeat(np.repeat(chunk[None], len(slots), axis=0), 256, axis=2)

        args = SimpleNamespace(
            max_steps=3,
            output_dir=Path(),
            mode="dagger-eval",
            vla_cadence=5,
            inference_batch_size=1,
        )
        with patch.object(haic_vla, "_new_pool", return_value=_FakePool()), patch.object(
            runtime, "canonical_state", return_value=torch.zeros(1, 605)
        ), patch.object(
            runtime,
            "refresh_rgb",
            return_value=np.zeros((1, 1, 1, 3), dtype=np.uint8),
        ), patch.object(runtime, "predict_vla", side_effect=predict_vla), patch.object(
            runtime, "student_actor_from_policy", return_value=RecordingActor()
        ):
            with tempfile.TemporaryDirectory() as output_dir:
                args.output_dir = Path(output_dir)
                result = haic_vla._dagger_eval(
                    args, env, SimpleNamespace(), SimpleNamespace()
                )

        self.assertEqual([int(item[0, 0].item()) for item in consumed], [0, 1, 2])
        self.assertEqual(result["motion_progress"]["mean"], 1.0)
        self.assertEqual(result["cart_progress"]["mean"], 1.0)
        self.assertEqual(result["pullcart_score"]["mean"], 100.0)

    def test_dagger_eval_clamps_cart_progress(self):
        class OutOfRangeCartEnv(_FakeEvalEnv):
            def step_and_maybe_reset(self, action):
                result = super().step_and_maybe_reset(action)
                if self.step_count == 1:
                    self.base_env.command_manager.object.data.root_link_pos_w[:, 0] = (
                        torch.tensor([-0.5, 1.5])
                    )
                return result

        env = OutOfRangeCartEnv([2, 2])
        args = SimpleNamespace(
            max_steps=2,
            output_dir=Path(),
            mode="dagger-eval",
            vla_cadence=1,
            inference_batch_size=2,
        )
        with patch.object(haic_vla, "_new_pool", return_value=_FakePool()), patch.object(
            runtime, "canonical_state", return_value=torch.zeros(2, 2)
        ), patch.object(
            runtime,
            "refresh_rgb",
            return_value=np.zeros((2, 1, 1, 3), dtype=np.uint8),
        ), patch.object(
            runtime,
            "predict_vla",
            return_value=np.zeros((2, 40, 256), dtype=np.float32),
        ), patch.object(
            runtime, "student_actor_from_policy", return_value=_FakeActor()
        ):
            with tempfile.TemporaryDirectory() as output_dir:
                args.output_dir = Path(output_dir)
                result = haic_vla._dagger_eval(
                    args, env, SimpleNamespace(), SimpleNamespace()
                )

        self.assertEqual(result["cart_progress"], {"mean": 0.5, "std": 0.5})
        self.assertEqual(result["pullcart_score"], {"mean": 50.0, "std": 25.0})

    def test_dagger_eval_does_not_write_partial_result(self):
        env = _FakeEvalEnv([1, 3])

        def predict_vla(_pool, _rgb, _state, slots, _step, _batch_size):
            return np.zeros((len(slots), 40, 256), dtype=np.float32)

        args = SimpleNamespace(
            max_steps=2,
            output_dir=Path(),
            mode="dagger-eval",
            vla_cadence=1,
            inference_batch_size=2,
        )
        with patch.object(haic_vla, "_new_pool", return_value=_FakePool()), patch.object(
            runtime, "canonical_state", return_value=torch.zeros(2, 2)
        ), patch.object(
            runtime,
            "refresh_rgb",
            return_value=np.zeros((2, 1, 1, 3), dtype=np.uint8),
        ), patch.object(
            runtime,
            "predict_vla",
            side_effect=predict_vla,
        ), patch.object(runtime, "student_actor_from_policy", return_value=_FakeActor()):
            with tempfile.TemporaryDirectory() as output_dir:
                args.output_dir = Path(output_dir)
                with self.assertRaisesRegex(RuntimeError, "1/2 episodes"):
                    haic_vla._dagger_eval(args, env, SimpleNamespace(), SimpleNamespace())
                self.assertFalse((args.output_dir / "dagger-eval.json").exists())

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

if __name__ == "__main__":
    unittest.main()
