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
        ),
    )
    parser.add_argument("--task", default=HAIC_TASK)
    parser.add_argument("--teacher-checkpoint", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
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
    parser.add_argument("--dagger-shard-rows", type=int, default=1_000)
    parser.add_argument("--config-dir", type=Path)
    AppLauncher.add_app_launcher_args(parser)
    return parser.parse_args()


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
            "10",
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


@torch.inference_mode()
def _bootstrap_collect(args, env, policy, simulation_app) -> dict[str, Any]:
    from active_adaptation.vla.runtime import (
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
    open_rows: list[dict[str, Any] | None] = [None for _ in range(env.num_envs)]
    rows_by_slot: list[list[dict[str, Any]]] = [[] for _ in range(env.num_envs)]
    frames_by_slot: list[list[np.ndarray]] = [[] for _ in range(env.num_envs)]
    terminal_by_slot = [False for _ in range(env.num_envs)]
    success_by_slot = [False for _ in range(env.num_envs)]

    while accepted < target_episodes and simulation_app.is_running():
        state = canonical_state(carry)
        teacher = teacher_action(policy, carry)
        due = [slot for slot, row in enumerate(open_rows) if row is None]
        if due:
            rgb = refresh_rgb(env)
            target = teacher_latent(policy, carry)
            for slot in due:
                open_rows[slot] = {
                    "state": state[slot].cpu().numpy(),
                    "action": target[slot].cpu().numpy(),
                    "actor_input": [],
                    "teacher_action": [],
                    "termination": [],
                }
                frames_by_slot[slot].append(rgb[slot].copy())

        for slot, row in enumerate(open_rows):
            if row is not None:
                row["actor_input"].append(state[slot].cpu().numpy())
                row["teacher_action"].append(teacher[slot].cpu().numpy())

        action_td = carry.clone(False)
        action_td["action"] = teacher
        td, carry = env.step_and_maybe_reset(action_td)
        done = td["next", "done"].squeeze(-1)
        success = td["next", "stats", "success"].squeeze(-1).bool()
        for slot, row in enumerate(open_rows):
            if row is None:
                continue
            row["termination"].append(bool(done[slot].item()))
            if done[slot].item():
                terminal_by_slot[slot] = True
                success_by_slot[slot] = bool(success[slot].item())
            if len(row["termination"]) != args.vla_cadence:
                continue

            rows_by_slot[slot].append(row)
            open_rows[slot] = None
            terminal = terminal_by_slot[slot]
            if not terminal:
                continue
            if success_by_slot[slot] and accepted < target_episodes:
                shard = output_dir / f".bootstrap_{episode_index:06d}.mp4"
                _encode_video(frames_by_slot[slot], shard)
                arrays = {
                    "state": np.stack([item["state"] for item in rows_by_slot[slot]]).astype(np.float32),
                    "action": np.stack([item["action"] for item in rows_by_slot[slot]]).astype(np.float32),
                    "actor_input": np.stack([item["actor_input"] for item in rows_by_slot[slot]]).astype(np.float32),
                    "teacher_action": np.stack([item["teacher_action"] for item in rows_by_slot[slot]]).astype(np.float32),
                    "termination": np.stack([item["termination"] for item in rows_by_slot[slot]]),
                    "image_shape": frames_by_slot[slot][0].shape,
                }
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
            frames_by_slot[slot] = []
            terminal_by_slot[slot] = False
            success_by_slot[slot] = False

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
            "schema_version": 3,
            "shard_format": "npz+mp4",
            "shard_root": "rollout_shards",
            "rows_unit": "decision_step",
            "env_fps": 50,
            "vla_fps": 10,
            "state_dim": 605,
            "vla_action_dim": 256,
            "teacher_latent_dim": 256,
            "actor_input_shape": [5, 605],
            "teacher_action_shape": [5, 23],
            "termination_shape": [5],
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
    latent = torch.zeros(env.num_envs, 256, device=env.device)
    rows: list[dict[str, Any]] = []
    row_count = 0
    shard_index = 0
    try:
        while row_count + len(rows) < args.row_budget and simulation_app.is_running():
            rgb = refresh_rgb(env)
            state = canonical_state(carry)
            slots = list(range(env.num_envs))
            predicted = predict_vla(
                pool,
                rgb,
                state.detach().cpu().numpy(),
                slots,
                row_count,
                args.inference_batch_size,
            )
            latent[:] = torch.from_numpy(predicted).to(env.device)
            target = teacher_latent(policy, carry)
            remaining = args.row_budget - row_count - len(rows)
            selected = slots[:remaining]
            row_data = {
                slot: {
                    "rgb": rgb[slot].copy(),
                    "state": state[slot].detach().cpu().numpy().astype(np.float32),
                    "action": target[slot].cpu().numpy().astype(np.float32),
                    "actor_input": [],
                    "teacher_action": [],
                    "termination": [],
                }
                for slot in selected
            }
            for _ in range(args.vla_cadence):
                pre = carry.clone(False)
                current_state = canonical_state(pre)
                labels = teacher_action(policy, pre)
                fixed_action = actor(torch.cat((current_state, latent), dim=-1))
                action_td = pre.clone(False)
                action_td["action"] = fixed_action
                td, carry = env.step_and_maybe_reset(action_td)
                done = td["next", "done"].squeeze(-1)
                for slot in selected:
                    row_data[slot]["actor_input"].append(
                        current_state[slot].cpu().numpy().astype(np.float32)
                    )
                    row_data[slot]["teacher_action"].append(
                        labels[slot].cpu().numpy().astype(np.float32)
                    )
                    row_data[slot]["termination"].append(bool(done[slot].item()))
                reset_ids = done.nonzero(as_tuple=False).flatten()
                if reset_ids.numel():
                    rgb_reset = refresh_rgb(env)
                    reset_state = canonical_state(carry)
                    reset_output = predict_vla(
                        pool,
                        rgb_reset,
                        reset_state.detach().cpu().numpy(),
                        reset_ids.cpu().tolist(),
                        row_count,
                        args.inference_batch_size,
                    )
                    latent[reset_ids] = torch.from_numpy(reset_output).to(env.device)
            for slot in selected:
                row_data[slot]["actor_input"] = np.asarray(
                    row_data[slot]["actor_input"], dtype=np.float32
                )
                row_data[slot]["teacher_action"] = np.asarray(
                    row_data[slot]["teacher_action"], dtype=np.float32
                )
                row_data[slot]["termination"] = np.asarray(row_data[slot]["termination"], dtype=bool)
                rows.append(row_data[slot])
            if len(rows) >= args.dagger_shard_rows:
                count = args.dagger_shard_rows
                _flush_dagger(output_dir, shard_index, rows[:count])
                rows = rows[count:]
                row_count += count
                shard_index += 1
    finally:
        if rows:
            _flush_dagger(output_dir, shard_index, rows)
            row_count += len(rows)
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
        "actor_input_shape": [5, 605],
        "teacher_action_shape": [5, 23],
        "termination_shape": [5],
    }
    _write_metadata(
        output_dir,
        {
            **result,
            "schema_version": 3,
            "shard_format": "npz+mp4",
            "shard_root": "rollout_shards/train",
            "rows_unit": "decision_step",
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
        canonical_state,
        predict_vla,
        refresh_rgb,
        student_actor_from_policy,
    )

    pool = _new_pool(args)
    actor = student_actor_from_policy(policy, env.device)
    env.base_env.eval()
    carry = env.reset()
    phase = torch.zeros(env.num_envs, dtype=torch.long, device=env.device)
    latent = torch.zeros(env.num_envs, 256, device=env.device)
    completed = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
    done_count = 0
    success_count = 0
    try:
        for step in range(args.max_steps):
            due = ((phase == 0) & ~completed).nonzero(as_tuple=False).flatten()
            if due.numel():
                rgb = refresh_rgb(env)
                state = canonical_state(carry)
                predicted = predict_vla(
                    pool,
                    rgb,
                    state.cpu().numpy(),
                    due.cpu().tolist(),
                    step,
                    args.inference_batch_size,
                )
                latent[due] = torch.from_numpy(predicted).to(env.device)
            action_td = carry.clone(False)
            action_td["action"] = actor(
                torch.cat((canonical_state(carry), latent), dim=-1)
            )
            td, carry = env.step_and_maybe_reset(action_td)
            done = td["next", "done"].squeeze(-1)
            success = td["next", "stats", "success"].squeeze(-1).bool()
            first_done = done & ~completed
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
    result = {
        "mode": args.mode,
        "episodes": done_count,
        "successes": success_count,
        "success_rate": success_count / done_count,
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
    phase = torch.zeros(env.num_envs, dtype=torch.long, device=env.device)
    latent = torch.zeros(env.num_envs, 256, device=env.device)
    completed = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
    done_count = 0
    success_count = 0
    for _ in range(args.max_steps):
        due = ((phase == 0) & ~completed).nonzero(as_tuple=False).flatten()
        if due.numel():
            target = teacher_latent(policy, carry)
            latent[due] = target[due]
        action_td = carry.clone(False)
        action_td["action"] = actor(torch.cat((canonical_state(carry), latent), dim=-1))
        td, carry = env.step_and_maybe_reset(action_td)
        done = td["next", "done"].squeeze(-1)
        success = td["next", "stats", "success"].squeeze(-1).bool()
        first_done = done & ~completed
        done_count += int(first_done.sum().item())
        success_count += int((first_done & success).sum().item())
        completed |= done
        phase.add_(1).remainder_(args.vla_cadence)
        phase[done] = 0
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
    from isaaclab.app import AppLauncher
    from omegaconf import OmegaConf

    args = _parse_args()
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
