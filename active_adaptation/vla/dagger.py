"""HAIC's native auto-reset, per-control-step DAgger adapter."""

import numpy as np
import torch

from active_adaptation.vla import runtime
from latency_bench.data.dagger import DaggerStep


class HaicDaggerAdapter:
    """Keep latent/actor labels and native chunk indexing at HAIC's control clock."""

    record_period = 1
    budget_mode = "episode"
    shard_rows = None
    seed_mode = "slot"
    reset = None

    def __init__(self, args, env, policy, write_episode):
        self.args, self.env, self.policy = args, env, policy
        self.num_envs = env.num_envs
        self.inference_period = args.vla_cadence
        self.actor = runtime.student_actor_from_policy(policy, env.device)
        self.carry = env.reset()
        self.write_episode = write_episode

    def snapshot(self, due, record_slots):
        rgb = runtime.refresh_rgb(self.env, update_hz=runtime.HAIC_CONTROL_HZ)
        state = runtime.canonical_state(self.carry)
        return {
            "rgb": rgb,
            "state": state,
            "state_cpu": state.cpu().numpy(),
        }

    def observations(self, snapshot, slots, state):
        return runtime.vla_observations(
            snapshot["rgb"][slots], snapshot["state_cpu"][slots], slots,
            [state.seeds[slot] for slot in slots], state.control_step,
        )

    def start_records(self, snapshot, slots, state):
        targets = runtime.teacher_latent(self.policy, self.carry).cpu().numpy()
        labels = runtime.teacher_action(self.policy, self.carry).cpu().numpy()
        return [
            {"rgb": snapshot["rgb"][slot].copy(),
             "state": snapshot["state_cpu"][slot].copy(),
             "action": targets[slot].copy(),
             "actor_input": snapshot["state_cpu"][slot].copy(),
             "teacher_action": labels[slot].copy(), "termination": False}
            for slot in slots
        ]

    def step(self, snapshot, active, state):
        latent = torch.from_numpy(np.stack([
            state.outputs[slot].action_chunk[state.episode_steps[slot] % self.inference_period]
            for slot in active
        ])).to(device=self.env.device, dtype=snapshot["state"].dtype)
        action = self.actor(torch.cat((snapshot["state"], latent), dim=-1))
        action_td = self.carry.clone(False)
        action_td["action"] = action
        td, self.carry = self.env.step_and_maybe_reset(action_td)
        done = td["next", "done"].squeeze(-1).nonzero(as_tuple=False).flatten().cpu().tolist()
        return DaggerStep(done, None)

    def finish_record(self, row, snapshot, result, slot, finished):
        row["termination"] = slot in result.done

    def write_rows(self, shard_index, slot, rows, writer):
        writer.submit(self.write_episode, self.args.output_dir.resolve(), shard_index, rows)
