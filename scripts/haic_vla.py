"""Collect and evaluate the HAIC PullCart GR00T VLA controller."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import traceback
from pathlib import Path
from typing import Any

import numpy as np
import torch

HAIC_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = Path(__file__).resolve().parents[3]
for source_root in (HAIC_ROOT, REPO_ROOT):
    source = str(source_root)
    if source in sys.path:
        sys.path.remove(source)
sys.path[:0] = [str(HAIC_ROOT), str(REPO_ROOT)]

from latency_bench.core.types import Action, Observation, StepResult
from latency_bench.envs.base import EnvAdapter
from latency_bench.envs.raw_rgb import ENV_RAW_RGB_FRAME_STACK_INFO_KEY


HAIC_TASK = "G1/haic/pull_cart"


def _parse_args() -> argparse.Namespace:
    from isaaclab.app import AppLauncher

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "mode",
        choices=(
            "bootstrap-collect",
            "dagger-collect",
            "dagger-eval",
            "oracle-eval",
            "profile",
        ),
    )
    parser.add_argument("--task", default=HAIC_TASK)
    parser.add_argument("--teacher-checkpoint", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--num-envs", type=int, default=8)
    parser.add_argument("--max-steps", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--split", choices=("train", "val"), default="train")
    parser.add_argument("--episode-budget", type=int, default=250)
    parser.add_argument("--policy-config", type=Path)
    parser.add_argument("--inference-device", default="cuda:0")
    parser.add_argument("--inference-batch-size", type=int, default=32)
    parser.add_argument("--dagger-round", type=int, default=0)
    parser.add_argument("--vla-cadence", type=int, default=5)
    parser.add_argument("--row-budget", type=int, default=32_000)
    parser.add_argument("--config-dir", type=Path)
    parser.add_argument("--profile-config", type=Path)
    AppLauncher.add_app_launcher_args(parser)
    args = parser.parse_args()
    if args.mode == "profile":
        if args.profile_config is None:
            parser.error("profile mode requires --profile-config")
    elif args.teacher_checkpoint is None or args.output_dir is None:
        parser.error("non-profile modes require --teacher-checkpoint and --output-dir")
    return args


def _compose_cfg(args: argparse.Namespace):
    import hydra
    import active_adaptation.learning.ppo.ppo_haic  # noqa: F401 - registers Hydra configs
    from omegaconf import OmegaConf

    config_dir = args.config_dir or Path(__file__).resolve().parents[1] / "cfg"
    overrides = [
        "algo=ppo_haic_train",
        f"task={args.task}",
        f"task.num_envs={args.num_envs}",
        "task.enable_cameras=false",
        "task.enable_vla_camera=true",
        "task.action.min_delay=0",
        "task.action.max_delay=0",
        "task.action.alpha=1.0",
        "app.enable_cameras=true",
        f"seed={args.seed}",
        f"checkpoint_path={args.teacher_checkpoint}",
        "vecnorm=eval",
        "eval_render=false",
    ]
    with hydra.initialize_config_dir(version_base=None, config_dir=str(config_dir.resolve())):
        cfg = hydra.compose(config_name="train", overrides=overrides)
    OmegaConf.resolve(cfg)
    OmegaConf.set_struct(cfg, False)
    return cfg


def _encode_video(frames: list[np.ndarray], path: Path) -> None:
    import imageio_ffmpeg

    frame = np.asarray(frames[0], dtype=np.uint8)
    process = subprocess.Popen(
        [
            imageio_ffmpeg.get_ffmpeg_exe(),
            "-loglevel",
            "error",
            "-y",
            "-f",
            "rawvideo",
            "-pixel_format",
            "rgb24",
            "-video_size",
            f"{frame.shape[1]}x{frame.shape[0]}",
            "-framerate",
            "50",
            "-i",
            "pipe:0",
            "-an",
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            str(path),
        ],
        stdin=subprocess.PIPE,
    )
    process.stdin.write(np.asarray(frames, dtype=np.uint8).tobytes())
    process.stdin.close()
    return_code = process.wait()
    if return_code:
        raise subprocess.CalledProcessError(return_code, process.args)


def _write_metadata(output_dir: Path, metadata: dict[str, Any]) -> None:
    temporary = output_dir / "metadata.json.tmp"
    temporary.write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(output_dir / "metadata.json")


def _new_pool(args: argparse.Namespace):
    from latency_bench.core.config import load_config
    from latency_bench.executors.realtime.pool import ProcessInferencePool

    config = load_config(args.policy_config)
    return ProcessInferencePool(
        config=config,
        inference_devices=[args.inference_device],
    )


class _HaicProfileEnv(EnvAdapter):
    """Adapt the live single-instance HAIC environment to the realtime executor."""

    env_fps = 50.0
    OBSERVATION_TYPE = "haic_pull_cart"

    def __init__(self, env, actor):
        self._env = env
        self._actor = actor
        self.noop_action = Action(
            value=np.zeros(256, dtype=np.float32),
            name="noop",
            is_noop=True,
        )
        self.env_step = 0
        self._carry = None

    def reset(self, seed: int | None = None) -> Observation:
        if seed is not None:
            self._env.set_seed(seed)
        self._carry = self._env.reset()
        self.env_step = 0
        return self.observe()

    def observe(self) -> Observation:
        from active_adaptation.vla.runtime import canonical_state, refresh_rgb

        state = canonical_state(self._carry)[0].detach().cpu().numpy()
        rgb = refresh_rgb(self._env, update_hz=10)[0]
        return Observation(
            data=None,
            env_step=self.env_step,
            sim_time_ms=self.env_step * 20.0,
            metadata={
                ENV_RAW_RGB_FRAME_STACK_INFO_KEY: rgb[None],
                "haic_state": state,
                "slot_id": 0,
            },
        )

    def step(self, action: Action) -> StepResult:
        from active_adaptation.vla.runtime import canonical_state

        latent = torch.as_tensor(
            action.value,
            device=self._env.device,
            dtype=canonical_state(self._carry).dtype,
        ).reshape(1, 256)
        actor_input = torch.cat((canonical_state(self._carry), latent), dim=-1)
        action_td = self._carry.clone(False)
        action_td["action"] = self._actor(actor_input)
        td, self._carry = self._env.step_and_maybe_reset(action_td)
        self.env_step += 1
        done = bool(td["next", "done"][0].item())
        truncated = bool(td["next", "truncated"][0].item())
        reward = float(td["next", "reward"][0].item())
        success = int(td["next", "stats", "success"][0].item())
        return StepResult(
            observation=None,
            reward=reward,
            done=done,
            truncated=truncated,
            info={"task_metrics": {"success": success}},
        )

    def render_game_frame(self):
        from active_adaptation.vla.runtime import rgb_frames

        return rgb_frames(self._env)[0]

    def close(self) -> None:
        self._env.close()


def _run_profile(profile_config: dict[str, Any], env, policy) -> dict[str, str]:
    from active_adaptation.vla.runtime import student_actor_from_policy
    from latency_bench.eval.driver import run_from_config

    actor = student_actor_from_policy(policy, env.device)
    profile_env = _HaicProfileEnv(
        env,
        actor,
    )
    run_from_config(
        profile_config,
        env=profile_env,
        inference_devices=profile_config["executor"]["inference_devices"],
    )
    return {"output_dir": profile_config["logging"]["output_dir"]}


@torch.inference_mode()
def _bootstrap_collect(args, env, policy, simulation_app) -> dict[str, Any]:
    from active_adaptation.vla.runtime import (
        HAIC_CONTROL_HZ,
        canonical_state,
        teacher_latent,
        refresh_rgb,
        teacher_action,
    )
    from latency_bench.data.haic_dagger import write_haic_bootstrap_shard

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    target_episodes = args.episode_budget
    accepted = 0
    episode_index = 0
    carry = env.reset()
    rows_by_slot: list[list[dict[str, Any]]] = [[] for _ in range(env.num_envs)]

    while accepted < target_episodes and simulation_app.is_running():
        state = canonical_state(carry)
        teacher = teacher_action(policy, carry)
        target = teacher_latent(policy, carry)
        rgb = refresh_rgb(env, update_hz=HAIC_CONTROL_HZ)
        state_cpu = state.detach().cpu().numpy().astype(np.float32, copy=False)
        teacher_cpu = teacher.detach().cpu().numpy().astype(np.float32, copy=False)
        target_cpu = target.detach().cpu().numpy().astype(np.float32, copy=False)
        for slot in range(env.num_envs):
            rows_by_slot[slot].append(
                {
                    "rgb": rgb[slot].copy(),
                    "state": state_cpu[slot].copy(),
                    "action": target_cpu[slot].copy(),
                    "actor_input": state_cpu[slot].copy(),
                    "teacher_action": teacher_cpu[slot].copy(),
                    "termination": False,
                }
            )

        action_td = carry.clone(False)
        action_td["action"] = teacher
        td, carry = env.step_and_maybe_reset(action_td)
        done = td["next", "done"].squeeze(-1)
        success = td["next", "stats", "success"].squeeze(-1).bool()
        for slot in done.nonzero(as_tuple=False).flatten().cpu().tolist():
            rows = rows_by_slot[slot]
            rows[-1]["termination"] = True
            if success[slot].item() and accepted < target_episodes:
                shard = output_dir / f".bootstrap_{episode_index:06d}.mp4"
                _encode_video([row["rgb"] for row in rows], shard)
                arrays = {
                    name: np.stack([item[name] for item in rows]).astype(np.float32)
                    for name in (
                        "state",
                        "action",
                        "actor_input",
                        "teacher_action",
                    )
                }
                arrays["termination"] = np.asarray(
                    [item["termination"] for item in rows], dtype=bool
                )
                arrays["image_shape"] = rows[0]["rgb"].shape
                write_haic_bootstrap_shard(
                    output_dir,
                    split=args.split,
                    episode_idx=episode_index,
                    arrays=arrays,
                    video_path=shard,
                )
                shard.unlink()
                accepted += 1
                episode_index += 1
            rows_by_slot[slot] = []
    if accepted != target_episodes:
        raise RuntimeError(
            f"bootstrap collector stopped at {accepted}/{target_episodes} episodes"
        )

    result = {
        "mode": args.mode,
        "accepted_episodes": accepted,
        "split": args.split,
        "episode_budget": args.episode_budget,
        "control_repeat": args.vla_cadence,
    }
    _write_metadata(
        output_dir,
        {
            **result,
            "schema_version": 4,
            "shard_format": "npz+mp4",
            "shard_root": "rollout_shards",
            "rows_unit": "control_step",
            "env_fps": 50,
            "vla_fps": 10,
            "state_dim": 605,
            "vla_action_dim": 256,
            "teacher_latent_dim": 256,
            "actor_input_shape": [605],
            "teacher_action_shape": [23],
            "termination_shape": [],
            "prompt": "Pull the cart along the reference motion.",
        },
    )
    (output_dir / "bootstrap-collect.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return result


def _flush_dagger(output_dir: Path, shard_index: int, rows: list[dict[str, Any]]) -> None:
    from latency_bench.data.haic_dagger import write_haic_dagger_shard

    video_path = output_dir / f".dagger_{shard_index:06d}.mp4"
    _encode_video([row["rgb"] for row in rows], video_path)
    arrays = {
        name: np.stack([row[name] for row in rows])
        for name in (
            "state",
            "action",
            "actor_input",
            "teacher_action",
            "termination",
        )
    }
    arrays["termination"][-1] = True
    arrays["image_shape"] = rows[0]["rgb"].shape
    write_haic_dagger_shard(
        output_dir,
        split="train",
        episode_idx=shard_index,
        arrays=arrays,
        video_path=video_path,
    )
    video_path.unlink()


@torch.inference_mode()
def _dagger_collect(args, env, policy, simulation_app) -> dict[str, Any]:
    from active_adaptation.vla.runtime import (
        HAIC_ACTION_HORIZON,
        HAIC_CONTROL_HZ,
        HAIC_LATENT_DIM,
        canonical_state,
        predict_vla,
        teacher_latent,
        refresh_rgb,
        teacher_action,
        student_actor_from_policy,
    )

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    pool = _new_pool(args)
    actor = student_actor_from_policy(policy, env.device)
    carry = env.reset()
    phase = torch.zeros(env.num_envs, dtype=torch.long, device=env.device)
    action_chunk = torch.zeros(
        env.num_envs, HAIC_ACTION_HORIZON, HAIC_LATENT_DIM, device=env.device
    )
    rows_by_slot: list[list[dict[str, Any]]] = [[] for _ in range(env.num_envs)]
    row_count = 0
    control_step = 0
    shard_index = 0
    try:
        while row_count < args.row_budget and simulation_app.is_running():
            rgb = refresh_rgb(env, update_hz=HAIC_CONTROL_HZ)
            state = canonical_state(carry)
            state_cpu = state.detach().cpu().numpy().astype(np.float32, copy=False)
            due = (phase == 0).nonzero(as_tuple=False).flatten()
            if due.numel():
                due_slots = due.cpu().tolist()
                predicted = predict_vla(
                    pool,
                    rgb[due_slots],
                    state_cpu[due.cpu().numpy()],
                    due_slots,
                    control_step,
                    args.inference_batch_size,
                )
                action_chunk[due] = torch.from_numpy(predicted).to(env.device)
            target = teacher_latent(policy, carry)
            target_cpu = target.cpu().numpy().astype(np.float32, copy=False)
            labels = teacher_action(policy, carry)
            labels_cpu = labels.cpu().numpy().astype(np.float32, copy=False)
            action_latent = action_chunk[
                torch.arange(env.num_envs, device=env.device), phase
            ]
            fixed_action = actor(torch.cat((state, action_latent), dim=-1))
            for slot in range(env.num_envs):
                rows_by_slot[slot].append(
                    {
                        "rgb": rgb[slot].copy(),
                        "state": state_cpu[slot].copy(),
                        "action": target_cpu[slot].copy(),
                        "actor_input": state_cpu[slot].copy(),
                        "teacher_action": labels_cpu[slot].copy(),
                        "termination": False,
                    }
                )
            action_td = carry.clone(False)
            action_td["action"] = fixed_action
            td, carry = env.step_and_maybe_reset(action_td)
            done = td["next", "done"].squeeze(-1)
            phase.add_(1).remainder_(args.vla_cadence)
            phase[done] = 0
            control_step += 1
            for slot in done.nonzero(as_tuple=False).flatten().cpu().tolist():
                episode_rows = rows_by_slot[slot]
                episode_rows[-1]["termination"] = True
                remaining = args.row_budget - row_count
                if remaining:
                    segment = episode_rows[:remaining]
                    _flush_dagger(output_dir, shard_index, segment)
                    row_count += len(segment)
                    shard_index += 1
                rows_by_slot[slot] = []
    finally:
        for episode_rows in rows_by_slot:
            remaining = args.row_budget - row_count
            if not episode_rows or not remaining:
                continue
            segment = episode_rows[:remaining]
            _flush_dagger(output_dir, shard_index, segment)
            row_count += len(segment)
            shard_index += 1
        pool.close()

    result = {
        "mode": args.mode,
        "round": args.dagger_round,
        "rows": row_count,
        "row_budget": args.row_budget,
        "control_repeat": args.vla_cadence,
        "state_dim": 605,
        "vla_action_dim": 256,
        "teacher_latent_dim": 256,
        "actor_input_shape": [605],
        "teacher_action_shape": [23],
        "termination_shape": [],
    }
    _write_metadata(
        output_dir,
        {
            **result,
            "schema_version": 4,
            "shard_format": "npz+mp4",
            "shard_root": "rollout_shards/train",
            "rows_unit": "control_step",
            "env_fps": 50,
            "vla_fps": 10,
            "prompt": "Pull the cart along the reference motion.",
        },
    )
    (output_dir / "dagger-collect.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return result


@torch.inference_mode()
def _dagger_eval(args, env, policy, simulation_app) -> dict[str, Any]:
    from active_adaptation.vla.runtime import (
        HAIC_ACTION_HORIZON,
        HAIC_LATENT_DIM,
        canonical_state,
        predict_vla,
        refresh_rgb,
        student_actor_from_policy,
    )

    pool = _new_pool(args)
    actor = student_actor_from_policy(policy, env.device)
    env.base_env.eval()
    carry = env.reset()
    command = env.base_env.command_manager
    motion_len = command.motion_len.clone()
    cart_start = command.object.data.root_link_pos_w.clone()
    ref_cart_positions = command.dataset.data.body_pos_w[
        torch.stack((command.motion_starts, command.motion_ends - 1)),
        command.object_body_id_motion,
    ]
    ref_cart_displacement = ref_cart_positions[1] - ref_cart_positions[0]
    phase = torch.zeros(env.num_envs, dtype=torch.long, device=env.device)
    action_chunk = torch.zeros(
        env.num_envs, HAIC_ACTION_HORIZON, HAIC_LATENT_DIM, device=env.device
    )
    completed = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
    motion_progress = torch.zeros(env.num_envs, device=env.device)
    cart_progress = torch.zeros(env.num_envs, device=env.device)
    done_count = 0
    success_count = 0
    try:
        for step in range(args.max_steps):
            due = ((phase == 0) & ~completed).nonzero(as_tuple=False).flatten()
            if due.numel():
                due_slots = due.cpu().tolist()
                rgb = refresh_rgb(env, due_slots)
                state = canonical_state(carry)
                predicted = predict_vla(
                    pool,
                    rgb,
                    state[due].cpu().numpy(),
                    due_slots,
                    step,
                    args.inference_batch_size,
                )
                action_chunk[due] = torch.from_numpy(predicted).to(env.device)
            action_latent = action_chunk[
                torch.arange(env.num_envs, device=env.device), phase
            ]
            action_td = carry.clone(False)
            action_td["action"] = actor(
                torch.cat((canonical_state(carry), action_latent), dim=-1)
            )
            motion_phase_before_step = command.t.clone()
            cart_position_before_step = command.object.data.root_link_pos_w.clone()
            td, carry = env.step_and_maybe_reset(action_td)
            done = td["next", "done"].squeeze(-1)
            success = td["next", "stats", "success"].squeeze(-1).bool()
            first_done = done & ~completed
            motion_progress[first_done] = (
                motion_phase_before_step[first_done].float()
                / (motion_len[first_done] - 1)
            )
            actual_cart_displacement = (
                cart_position_before_step[first_done] - cart_start[first_done]
            )
            target_cart_displacement = ref_cart_displacement[first_done]
            cart_progress[first_done] = (
                (actual_cart_displacement * target_cart_displacement).sum(dim=-1)
                / target_cart_displacement.square().sum(dim=-1)
            ).clamp(0.0, 1.0)
            done_count += int(first_done.sum().item())
            success_count += int((first_done & success).sum().item())
            completed |= done
            phase.add_(1).remainder_(args.vla_cadence)
            phase[done] = 0
            if completed.all():
                break
    finally:
        pool.close()
    if not completed.all():
        raise RuntimeError(
            f"dagger evaluator stopped at {done_count}/{env.num_envs} episodes"
        )
    pullcart_score = 50.0 * (motion_progress + cart_progress)

    def summary(values: torch.Tensor) -> dict[str, float]:
        return {
            "mean": float(values.mean().item()),
            "std": float(values.std(correction=0).item()),
        }

    result = {
        "mode": args.mode,
        "episodes": done_count,
        "successes": success_count,
        "success_rate": success_count / done_count,
        "motion_progress": summary(motion_progress),
        "cart_progress": summary(cart_progress),
        "pullcart_score": summary(pullcart_score),
    }
    (args.output_dir / "dagger-eval.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return result


@torch.inference_mode()
def _oracle_eval(args, env, policy, simulation_app) -> dict[str, Any]:
    from active_adaptation.vla.runtime import (
        canonical_state,
        student_actor_from_policy,
        teacher_latent,
    )

    actor = student_actor_from_policy(policy, env.device)
    env.base_env.eval()
    carry = env.reset()
    completed = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
    done_count = 0
    success_count = 0
    for _ in range(args.max_steps):
        target = teacher_latent(policy, carry)
        action_td = carry.clone(False)
        action_td["action"] = actor(torch.cat((canonical_state(carry), target), dim=-1))
        td, carry = env.step_and_maybe_reset(action_td)
        done = td["next", "done"].squeeze(-1)
        success = td["next", "stats", "success"].squeeze(-1).bool()
        first_done = done & ~completed
        done_count += int(first_done.sum().item())
        success_count += int((first_done & success).sum().item())
        completed |= done
        if completed.all():
            break
    if not completed.all():
        raise RuntimeError(
            f"oracle evaluator stopped at {done_count}/{env.num_envs} episodes"
        )
    result = {
        "mode": args.mode,
        "episodes": done_count,
        "successes": success_count,
        "success_rate": success_count / done_count,
    }
    (args.output_dir / "oracle-eval.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return result


def main() -> None:
    from latency_bench.core.config import load_config
    from isaaclab.app import AppLauncher
    from omegaconf import OmegaConf

    args = _parse_args()
    profile_config = None
    if args.mode == "profile":
        profile_config = load_config(args.profile_config)
        args.teacher_checkpoint = Path(profile_config["env"]["runtime_checkpoint_path"])
        args.output_dir = Path(profile_config["logging"]["output_dir"])
        args.num_envs = 1
        args.max_steps = profile_config["evaluation"]["eval_max_steps"]
        args.seed = profile_config["experiment"]["seed"]
        args.device = profile_config["env"]["simulator_device"]

    args.output_dir.mkdir(parents=True, exist_ok=True)
    cfg = _compose_cfg(args)
    app_launcher = AppLauncher(OmegaConf.to_container(cfg.app), device=args.device)
    simulation_app = app_launcher.app
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from helpers import make_env_policy

    env, policy, _ = make_env_policy(cfg)
    env.eval()
    policy.eval()
    exit_code = 0
    try:
        if args.mode == "bootstrap-collect":
            result = _bootstrap_collect(args, env, policy, simulation_app)
        elif args.mode == "dagger-collect":
            result = _dagger_collect(args, env, policy, simulation_app)
        elif args.mode == "profile":
            result = _run_profile(profile_config, env, policy)
        elif args.mode == "dagger-eval":
            result = _dagger_eval(args, env, policy, simulation_app)
        else:
            result = _oracle_eval(args, env, policy, simulation_app)
        print(json.dumps(result, indent=2, sort_keys=True))
    except BaseException:
        traceback.print_exc()
        exit_code = 1
    finally:
        env.close()
        simulation_app.close(skip_cleanup=True)
    if exit_code:
        raise SystemExit(exit_code)


if __name__ == "__main__":
    main()
