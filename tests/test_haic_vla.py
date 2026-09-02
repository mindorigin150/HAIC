from __future__ import annotations

import unittest

import torch

from active_adaptation.learning.ppo.haic_actor import (
    HaicStudentActor,
    _ACTOR_ADAPT_KEYS,
    extract_actor_adapt_state_dict,
)


class HaicVlaContractTest(unittest.TestCase):
    def test_student_actor_tensor_contract_and_gradients(self):
        actor = HaicStudentActor()
        actor_input = torch.zeros(2, 861)
        loc = actor(actor_input)
        self.assertEqual(loc.shape, (2, 23))
        loc.square().mean().backward()

    def test_native_actor_adapt_key_extraction(self):
        actor = HaicStudentActor()
        native_state = {
            source_key: actor.state_dict()[target_key]
            for target_key, source_key in _ACTOR_ADAPT_KEYS.items()
        }
        actor.load_state_dict(extract_actor_adapt_state_dict(native_state))

if __name__ == "__main__":
    unittest.main()
