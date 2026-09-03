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
from active_adaptation.vla.runtime import student_actor_from_policy, teacher_latent
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
        self.base_env = SimpleNamespace(eval=lambda: self.events.append("eval"))

    def reset(self):
        self.events.append("reset")
        return _FakeTensorDict()

    def step_and_maybe_reset(self, _action):
        self.step_count += 1
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

    def test_dagger_flush_preserves_action_distillation_window(self):
        row = {
            "rgb": np.zeros((2, 2, 3), dtype=np.uint8),
            "state": np.zeros(605, dtype=np.float32),
            "action": np.zeros(256, dtype=np.float32),
            "actor_input": np.zeros((5, 605), dtype=np.float32),
            "teacher_action": np.zeros((5, 23), dtype=np.float32),
            "termination": np.asarray([False, False, False, False, True]),
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
                self.assertEqual(payload["actor_input"].shape, (1, 5, 605))
                self.assertEqual(payload["teacher_action"].shape, (1, 5, 23))
                self.assertEqual(payload["termination"].shape, (1, 5))

    def test_dagger_eval_counts_only_first_done_per_env(self):
        env = _FakeEvalEnv([1, 2])
        prediction_slots = []

        def predict_vla(_pool, _rgb, _state, slots, _step, _batch_size):
            prediction_slots.append(tuple(slots))
            return np.zeros((len(slots), 256), dtype=np.float32)

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
                self.assertNotIn("steps", result)
                self.assertEqual(prediction_slots, [(0, 1), (1,)])
                self.assertEqual(env.events[:2], ["eval", "reset"])

    def test_dagger_eval_does_not_write_partial_result(self):
        env = _FakeEvalEnv([1, 3])

        def predict_vla(_pool, _rgb, _state, slots, _step, _batch_size):
            return np.zeros((len(slots), 256), dtype=np.float32)

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
