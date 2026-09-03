"""Runtime primitives shared by the HAIC VLA collector and evaluator."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np
import torch

from active_adaptation.learning.ppo.haic_actor import (
    HaicStudentActor,
    extract_actor_adapt_state_dict,
)


HAIC_VLA_HZ = 10
HAIC_CONTROL_HZ = 50
HAIC_ACTION_HORIZON = 40
HAIC_LATENT_DIM = 256


def canonical_state(tensordict) -> torch.Tensor:
    """Return the exact normalized command+policy deployment state."""
    return torch.cat((tensordict["command"], tensordict["policy"]), dim=-1)


def _teacher_encoded(policy, tensordict):
    encoded = tensordict.clone(False)
    policy.object_transform(encoded)
    policy.encoder_priv(encoded)
    return encoded


@torch.inference_mode()
def teacher_latent(policy, tensordict) -> torch.Tensor:
    """Return the 256D privileged latent consumed by the HAIC actor."""
    encoded = _teacher_encoded(policy, tensordict)
    return encoded["priv_feature"]


@torch.inference_mode()
def teacher_action(policy, tensordict) -> torch.Tensor:
    """Return the full teacher distribution mean after reference residual."""
    encoded = _teacher_encoded(policy, tensordict)
    return policy.actor.get_dist(encoded).mean


def student_actor_from_policy(policy, device: torch.device) -> HaicStudentActor:
    """Copy native ``policy.actor_adapt`` weights into the pure actor."""
    actor = HaicStudentActor().to(device)
    actor.load_state_dict(
        extract_actor_adapt_state_dict(policy.actor_adapt.state_dict())
    )
    actor.requires_grad_(False).eval()
    return actor


def rgb_frames(env, slot_ids: list[int] | None = None) -> np.ndarray:
    """Read the latest RGB frames, compacting partial reads in slot order."""
    camera = env.base_env.scene["vla_camera"]
    output = camera.data.output["rgb"]
    if slot_ids is None:
        return output[..., :3].detach().cpu().numpy().astype(np.uint8, copy=False)

    return output[slot_ids][..., :3].detach().cpu().numpy().astype(
        np.uint8, copy=False
    )


def refresh_rgb(
    env,
    slot_ids: list[int] | None = None,
    *,
    update_hz: int = HAIC_VLA_HZ,
) -> np.ndarray:
    """Render all cameras, then transfer only requested slots to the host."""
    env.base_env.sim.render()
    env.base_env.scene["vla_camera"].update(
        1.0 / update_hz,
        force_recompute=True,
    )
    return rgb_frames(env, slot_ids)


def vla_observations(
    rgb: np.ndarray,
    state: np.ndarray,
    slots: Sequence[int],
    control_step: int,
) -> list[Any]:
    """Build parent-process observations for the official GR00T pool."""
    from latency_bench.core.types import Observation
    from latency_bench.envs.raw_rgb import ENV_RAW_RGB_FRAME_STACK_INFO_KEY

    return [
        Observation(
            data=None,
            env_step=control_step,
            sim_time_ms=control_step * 20.0,
            metadata={
                ENV_RAW_RGB_FRAME_STACK_INFO_KEY: frame[None],
                "haic_state": state_value,
                "slot_id": slot,
            },
        )
        for frame, state_value, slot in zip(rgb, state, slots)
    ]


def predict_vla(
    pool,
    rgb: np.ndarray,
    state: np.ndarray,
    slots: Sequence[int],
    step: int,
    batch_size: int,
) -> np.ndarray:
    """Predict one GR00T action chunk per requested environment slot."""
    actions = []
    for start in range(0, len(slots), batch_size):
        batch = slice(start, start + batch_size)
        observations = vla_observations(
            rgb[batch], state[batch], slots[batch], step
        )
        actions.extend(
            np.asarray(output.action_chunk, dtype=np.float32)
            for output in pool.predict_batch(observations)
        )
    return np.stack(actions)
