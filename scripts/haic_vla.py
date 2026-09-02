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
        choices=("bootstrap-collect", "dagger-collect", "dagger-eval"),
    )
    parser.add_argument("--task", default=HAIC_TASK)
    parser.add_argument("--teacher-checkpoint", required=True)
    parser.add_argument("--student-actor-checkpoint", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--num-envs", type=int, default=8)
    parser.add_argument("--max-steps", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--split", choices=("train", "val"), default="train")
    parser.add_argument("--episode-budget", type=int, default=250)
    parser.add_argument("--policy-config", type=Path)
    parser.add_argument("--inference-device", default="cuda:0")
    parser.add_argument("--inference-batch-size", type=int, default=8)
    parser.add_argument("--dagger-round", type=int, default=0)
    parser.add_argument("--vla-cadence", type=int, default=5)
    parser.add_argument("--row-budget", type=int, default=2_304_000)
    parser.add_argument("--dagger-shard-rows", type=int, default=256)
    parser.add_argument("--config-dir", type=Path)
    AppLauncher.add_app_launcher_args(parser)
    return parser.parse_args()


def _compose_cfg(args: argparse.Namespace):
    import hydra
    import active_adaptation.learning.ppo.ppo_haic  # register Hydra configs
    from omegaconf import OmegaConf

    config_dir = args.config_dir or Path(__file__).resolve().parents[1] / "cfg"
    overrides = [
        "algo=ppo_haic_train",
        f"task={args.task}",
        f"task.num_envs={args.num_envs}",
        "task.enable_cameras=false",
        "task.enable_vla_camera=true",
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


def _load_student_actor(policy, checkpoint: Path | None, device: torch.device):
    from active_adaptation.learning.ppo.haic_actor import HaicStudentActor
    from active_adaptation.vla.runtime import student_actor_from_policy

    if checkpoint is None:
        return student_actor_from_policy(policy, device)
    actor = HaicStudentActor().to(device)
    actor.load_state_dict(torch.load(checkpoint, map_location=device, weights_only=True))
    actor.eval()
    return actor


@torch.inference_mode()
def _bootstrap_collect(args, env, policy, simulation_app) -> dict[str, Any]:
    from active_adaptation.vla.runtime import (
        canonical_state,
        privileged_target,
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
    phase = torch.zeros(env.num_envs, dtype=torch.long, device=env.device)
    rows_by_slot: list[list[dict[str, Any]]] = [[] for _ in range(env.num_envs)]
    frames_by_slot: list[list[np.ndarray]] = [[] for _ in range(env.num_envs)]

    while accepted < target_episodes and simulation_app.is_running():
        due = (phase == 0).nonzero(as_tuple=False).flatten()
        state = canonical_state(carry)
        if due.numel():
            rgb = refresh_rgb(env)
            target = privileged_target(policy, carry)
            for slot in due.cpu().tolist():
                rows_by_slot[slot].append(
                    {"state": state[slot].cpu().numpy(), "action": target[slot].cpu().numpy()}
                )
                frames_by_slot[slot].append(rgb[slot].copy())

        action = teacher_action(policy, carry)
        action_td = carry.clone(False)
        action_td["action"] = action
        td, carry = env.step_and_maybe_reset(action_td)
        phase.add_(1).remainder_(args.vla_cadence)
        done = td["next", "done"].squeeze(-1)
        success = td["next", "stats", "success"].squeeze(-1).bool()
        for slot in done.nonzero(as_tuple=False).flatten().cpu().tolist():
            if success[slot].item() and accepted < target_episodes:
                shard = output_dir / f".bootstrap_{episode_index:06d}.mp4"
                _encode_video(frames_by_slot[slot], shard)
                arrays = {
                    "state": np.stack([row["state"] for row in rows_by_slot[slot]]).astype(np.float32),
                    "action": np.stack([row["action"] for row in rows_by_slot[slot]]).astype(np.float32),
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
            phase[slot] = 0

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
            "schema_version": 1,
            "shard_format": "npz+mp4",
            "shard_root": "rollout_shards",
            "rows_unit": "decision_step",
            "env_fps": 50,
            "vla_fps": 10,
            "state_dim": 605,
            "vla_action_dim": 275,
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
        for name in ("state", "action", "actor_input", "teacher_action", "termination", "is_init")
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
        privileged_target,
        refresh_rgb,
        teacher_action,
    )

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    pool = _new_pool(args)
    student = _load_student_actor(policy, args.student_actor_checkpoint, env.device)
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
            latent[:] = torch.from_numpy(predicted[:, :256]).to(env.device)
            target = privileged_target(policy, carry)
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
                    "is_init": [],
                    "after_termination": False,
                }
                for slot in selected
            }
            for _ in range(args.vla_cadence):
                pre = carry.clone(False)
                current_state = canonical_state(pre)
                labels = teacher_action(policy, pre)
                student_action = student(torch.cat((current_state, latent), dim=-1))
                action_td = pre.clone(False)
                action_td["action"] = student_action
                td, carry = env.step_and_maybe_reset(action_td)
                done = td["next", "done"].squeeze(-1)
                for slot in selected:
                    row_data[slot]["actor_input"].append(current_state[slot].cpu().numpy())
                    row_data[slot]["teacher_action"].append(labels[slot].cpu().numpy())
                    row_data[slot]["termination"].append(bool(done[slot].item()))
                    row_data[slot]["is_init"].append(
                        row_data[slot]["after_termination"]
                        or bool(pre["is_init"][slot].item())
                    )
                    row_data[slot]["after_termination"] |= bool(done[slot].item())
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
                    latent[reset_ids] = torch.from_numpy(reset_output[:, :256]).to(env.device)
            for slot in selected:
                del row_data[slot]["after_termination"]
                row_data[slot]["actor_input"] = np.asarray(row_data[slot]["actor_input"], dtype=np.float32)
                row_data[slot]["teacher_action"] = np.asarray(row_data[slot]["teacher_action"], dtype=np.float32)
                row_data[slot]["termination"] = np.asarray(row_data[slot]["termination"], dtype=bool)
                row_data[slot]["is_init"] = np.asarray(row_data[slot]["is_init"], dtype=bool)
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
        "vla_action_dim": 275,
        "teacher_action_dim": 23,
    }
    _write_metadata(
        output_dir,
        {
            **result,
            "schema_version": 1,
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
    from active_adaptation.vla.runtime import canonical_state, predict_vla, refresh_rgb

    pool = _new_pool(args)
    student = _load_student_actor(policy, args.student_actor_checkpoint, env.device)
    carry = env.reset()
    phase = torch.zeros(env.num_envs, dtype=torch.long, device=env.device)
    latent = torch.zeros(env.num_envs, 256, device=env.device)
    done_count = 0
    success_count = 0
    try:
        for step in range(args.max_steps):
            due = (phase == 0).nonzero(as_tuple=False).flatten()
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
                latent[due] = torch.from_numpy(predicted[:, :256]).to(env.device)
            action_td = carry.clone(False)
            action_td["action"] = student(
                torch.cat((canonical_state(carry), latent), dim=-1)
            )
            td, carry = env.step_and_maybe_reset(action_td)
            done = td["next", "done"].squeeze(-1)
            success = td["next", "stats", "success"].squeeze(-1).bool()
            done_count += int(done.sum().item())
            success_count += int((done & success).sum().item())
            phase.add_(1).remainder_(args.vla_cadence)
            phase[done] = 0
    finally:
        pool.close()
    result = {
        "mode": args.mode,
        "steps": args.max_steps,
        "episodes": done_count,
        "successes": success_count,
        "success_rate": success_count / done_count,
    }
    (args.output_dir / "dagger-eval.json").write_text(
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
        else:
            result = _dagger_eval(args, env, policy, simulation_app)
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
