from __future__ import annotations

import unittest
from types import SimpleNamespace

import torch

from active_adaptation.learning.ppo.haic_actor import (
    HaicStudentActor,
    _ACTOR_ADAPT_KEYS,
    extract_actor_adapt_state_dict,
)
from active_adaptation.vla.runtime import student_actor_from_policy, teacher_latent


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

if __name__ == "__main__":
    unittest.main()
