"""Latency-bench environment backend for the HAIC PullCart task.

The Isaac/TorchRL environment remains task-owned.  This module only adapts
its vector reset/step/observation contract to latency_bench's slot backend;
policy inference and latency scheduling stay in the common evaluator.
"""

from __future__ import annotations

import os
import time
from collections.abc import Mapping, Sequence
from typing import Any

import gymnasium as gym
import numpy as np
import torch

from latency_bench.core.types import Action, Observation, StepResult
from latency_bench.executors.env_step_backend import (
    EnvStepResponse,
    RemoteEnvSlotHandle,
)
from latency_bench.envs.raw_rgb import ENV_RAW_RGB_FRAME_STACK_INFO_KEY

from .runtime import (
    HAIC_LATENT_DIM,
    canonical_state,
    refresh_rgb,
    student_actor_from_policy,
)


class HaicEnvStepBackend:
    """Run a transformed HAIC vector environment without automatic reset.

    ``TransformedEnv.step_and_maybe_reset`` is intentionally not used here:
    the common latency evaluator owns episode boundaries and calls
    :meth:`reset_slot` after it has recorded a terminal step.
    """

    backend_name = "haic_vector"

    def __init__(self, env, actor):
        self.env = env
        self.base_env = env.base_env
        self.actor = actor
        self._carry = None
        self._env_steps = np.zeros(env.num_envs, dtype=np.int64)
        self._episode_ids: list[int | None] = [None] * env.num_envs
        self._episode_seeds: list[int | None] = [None] * env.num_envs
        self._motion_len: list[torch.Tensor | None] = [None] * env.num_envs
        self._cart_start: list[torch.Tensor | None] = [None] * env.num_envs
        self._ref_cart_displacement: list[torch.Tensor | None] = [None] * env.num_envs
        self.noop_action = Action(
            value=np.zeros(HAIC_LATENT_DIM, dtype=np.float32),
            name="noop",
            is_noop=True,
        )
        action_space = gym.spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(HAIC_LATENT_DIM,),
            dtype=np.float32,
        )
        self.slot_handles = [
            RemoteEnvSlotHandle(
                slot_id=slot_id,
                env_fps=1.0 / float(env.base_env.step_dt),
                noop_action=self.noop_action,
                action_space=action_space,
            )
            for slot_id in range(env.num_envs)
        ]
        self.last_step_metadata_by_slot: dict[int, dict[str, Any]] = {}
        self.closed = False

    @property
    def num_slots(self) -> int:
        return len(self.slot_handles)

    @property
    def env_fps(self) -> float:
        return float(self.slot_handles[0].env_fps)

    def worker_pids(self) -> list[int]:
        return []

    def reset_slot(self, slot_id: int, *, episode_id: int, seed: int | None) -> Observation:
        slot_id = int(slot_id)
        if self._carry is None:
            self._carry = self.env.reset()
        self._reset_slot_state(slot_id, seed)
        self._episode_ids[slot_id] = int(episode_id)
        self._episode_seeds[slot_id] = None if seed is None else int(seed)
        self._env_steps[slot_id] = 0
        observation = self._observation(slot_id)
        self.slot_handles[slot_id].update_observation(observation)
        return observation

    def observe_slots(self, slot_ids: Sequence[int]) -> dict[int, Observation]:
        slot_ids = [int(slot_id) for slot_id in slot_ids]
        frames = refresh_rgb(self.env)
        state = canonical_state(self._carry).detach().cpu().numpy()
        observations = {
            slot_id: self._make_observation(slot_id, frames[slot_id], state[slot_id])
            for slot_id in slot_ids
        }
        for slot_id, observation in observations.items():
            self.slot_handles[slot_id].update_observation(observation)
        return observations

    def step_slots(self, actions_by_slot: Mapping[int, Action]) -> dict[int, EnvStepResponse]:
        slot_ids = [int(slot_id) for slot_id in actions_by_slot]
        action_values = torch.zeros(
            self.num_slots,
            HAIC_LATENT_DIM,
            dtype=canonical_state(self._carry).dtype,
            device=self.base_env.device,
        )
        for slot_id, action in actions_by_slot.items():
            action_values[int(slot_id)] = torch.as_tensor(
                action.value,
                dtype=action_values.dtype,
                device=action_values.device,
            ).reshape(HAIC_LATENT_DIM)

        command = self.base_env.command_manager
        state_before = canonical_state(self._carry)
        motion_phase_before = command.t.clone()
        cart_position_before = command.object.data.root_link_pos_w.clone()
        action_td = self._carry.clone(False)
        action_td["action"] = self.actor(torch.cat((state_before, action_values), dim=-1))

        start = time.perf_counter()
        td = self.env.step(action_td)
        from torchrl.envs.utils import step_mdp

        self._carry = step_mdp(td)
        elapsed = time.perf_counter() - start
        next_td = td["next"]
        done = next_td["done"].squeeze(-1)
        truncated = next_td["truncated"].squeeze(-1)
        reward = next_td["reward"].reshape(self.num_slots, -1)[:, 0]
        success = next_td["stats", "success"].reshape(self.num_slots, -1)[:, 0]
        responses: dict[int, EnvStepResponse] = {}
        for slot_id in slot_ids:
            self._env_steps[slot_id] += 1
            info = {
                "env_step": int(self._env_steps[slot_id]),
                "sim_time_ms": int(self._env_steps[slot_id]) * self.slot_handles[slot_id].frame_ms,
                "applied_action": action_values[slot_id].detach().cpu().tolist(),
                "task_metrics": {"success": int(success[slot_id].item())},
            }
            if bool(done[slot_id].item()):
                motion_len = self._motion_len[slot_id]
                cart_start = self._cart_start[slot_id]
                ref_displacement = self._ref_cart_displacement[slot_id]
                motion_progress = motion_phase_before[slot_id].float() / (motion_len - 1)
                actual_displacement = cart_position_before[slot_id] - cart_start
                cart_progress = (
                    (actual_displacement * ref_displacement).sum()
                    / ref_displacement.square().sum()
                ).clamp(0.0, 1.0)
                motion_progress = float(motion_progress.item())
                cart_progress = float(cart_progress.item())
                info["task_metrics"].update(
                    {
                        "motion_progress": motion_progress,
                        "cart_progress": cart_progress,
                        "pullcart_score": 50.0 * (motion_progress + cart_progress),
                    }
                )
            result = StepResult(
                observation=None,
                reward=float(reward[slot_id].item()),
                done=bool(done[slot_id].item()),
                truncated=bool(truncated[slot_id].item()),
                info=info,
            )
            metadata = {
                "backend": self.backend_name,
                "slot_id": slot_id,
                "worker_pid": os.getpid(),
                "worker_step_wall_sec": elapsed,
            }
            self.last_step_metadata_by_slot[slot_id] = metadata
            responses[slot_id] = EnvStepResponse(
                result=result,
                worker_pid=os.getpid(),
                metadata=metadata,
            )
        return responses

    def close(self) -> None:
        if self.closed:
            return
        self.closed = True
        self.env.close()

    def _reset_slot_state(self, slot_id: int, seed: int | None) -> None:
        self.base_env.set_episode_seed(slot_id, seed)
        reset_mask = torch.zeros(self.num_slots, dtype=torch.bool, device=self.base_env.device)
        reset_mask[slot_id] = True
        reset_td = self._carry.clone(False)
        reset_td["_reset"] = reset_mask
        self._carry = self.env.reset(reset_td)

        command = self.base_env.command_manager
        self._motion_len[slot_id] = command.motion_len[slot_id].clone()
        self._cart_start[slot_id] = command.object.data.root_link_pos_w[slot_id].clone()
        reference_positions = command.dataset.data.body_pos_w[
            torch.stack((command.motion_starts, command.motion_ends - 1)),
            command.object_body_id_motion,
        ]
        self._ref_cart_displacement[slot_id] = (
            reference_positions[1, slot_id] - reference_positions[0, slot_id]
        ).clone()

    def _observation(self, slot_id: int) -> Observation:
        frames = refresh_rgb(self.env)
        state = canonical_state(self._carry).detach().cpu().numpy()
        return self._make_observation(slot_id, frames[slot_id], state[slot_id])

    def _make_observation(self, slot_id: int, frame: np.ndarray, state: np.ndarray) -> Observation:
        env_step = int(self._env_steps[slot_id])
        return Observation(
            data=None,
            env_step=env_step,
            sim_time_ms=env_step * self.slot_handles[slot_id].frame_ms,
            metadata={
                ENV_RAW_RGB_FRAME_STACK_INFO_KEY: frame[None],
                "haic_state": state,
                "slot_id": slot_id,
                "episode_id": self._episode_ids[slot_id],
                "episode_seed": self._episode_seeds[slot_id],
            },
        )


def build_haic_env_backend(env, policy) -> HaicEnvStepBackend:
    return HaicEnvStepBackend(env, student_actor_from_policy(policy, env.device))
