"""Collect and evaluate the HAIC PullCart GR00T VLA controller."""

from __future__ import annotations

import argparse
import json
import os
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

HAIC_TASK = "G1/haic/pull_cart"


def _parse_args() -> argparse.Namespace:
    from isaaclab.app import AppLauncher

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "mode",
        choices=(
            "bootstrap-collect",
            "teacher-collect",
            "dagger-collect",
            "oracle-eval",
            "latency-eval",
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
    parser.add_argument("--keep-failed", action="store_true")
    parser.add_argument("--policy-config", type=Path)
    parser.add_argument("--latency-config", type=Path)
    parser.add_argument("--inference-device", default="cuda:0")
    parser.add_argument("--inference-batch-size", type=int, default=32)
    parser.add_argument("--dagger-round", type=int, default=0)
    parser.add_argument("--vla-cadence", type=int, default=5)
    parser.add_argument("--action-horizon", type=int, default=40)
    parser.add_argument("--row-budget", type=int, default=32_000)
    parser.add_argument("--config-dir", type=Path)
    parser.add_argument("--eval-config", type=Path)
    AppLauncher.add_app_launcher_args(parser)
    args = parser.parse_args()
    if args.mode == "latency-eval":
        if args.eval_config is None:
            parser.error("latency-eval mode requires --eval-config")
    elif args.teacher_checkpoint is None or args.output_dir is None:
        parser.error("non-latency-eval modes require --teacher-checkpoint and --output-dir")
    elif args.mode == "bootstrap-collect" and args.latency_config is None:
        parser.error("bootstrap-collect mode requires --latency-config")
    return args


def _compose_cfg(args: argparse.Namespace):
    import hydra
    import active_adaptation.learning.ppo.ppo_haic  # noqa: F401 - registers Hydra configs
    from omegaconf import OmegaConf

    config_dir = args.config_dir or Path(__file__).resolve().parents[1] / "cfg"
    algo = (
        "ppo_haic_latency_command"
        if args.mode == "bootstrap-collect"
        else "ppo_haic_train"
    )
    overrides = [
        f"algo={algo}",
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
    if args.mode == "bootstrap-collect":
        overrides.extend(
            [
                "task.latency_command=true",
                f"task.latency_config_path={args.latency_config}",
                "task.latency_control_repeat=1",
            ]
        )
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


def _run_latency_eval(eval_config: dict[str, Any], env, policy) -> dict[str, str]:
    from active_adaptation.vla.backend import build_haic_env_backend
    from latency_bench.eval.driver import run_from_config

    env_backend = build_haic_env_backend(env, policy, obs_fps=eval_config["env"]["obs_fps"])
    run_from_config(
        eval_config,
        env_backend=env_backend,
        # The task process keeps the native actor-adapt copy for the
        # 256D-latent -> motor-action transform.  The common executor owns
        # policy inference (in-process for simulated eval, worker process for
        # realtime eval) and must therefore build it from the eval config.
        policy=None,
        inference_devices=eval_config["executor"]["inference_devices"],
    )
    return {"output_dir": eval_config["logging"]["output_dir"]}


@torch.inference_mode()
def _bootstrap_collect(args, env, policy, simulation_app) -> dict[str, Any]:
    """Collect admitted native command chunks with raw-frame decoder labels."""
    from active_adaptation.vla.runtime import (
        HAIC_ACTION_HORIZON,
        HAIC_CONTROL_HZ,
        HAIC_LATENT_DIM,
        canonical_state,
        teacher_command,
        refresh_rgb,
        student_actor_from_policy,
    )
    from latency_bench.data.haic_dagger import write_haic_bootstrap_shard

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    transport = env.base_env.command_latency
    decoder = student_actor_from_policy(policy, env.device)
    target_episodes = args.episode_budget
    accepted = 0
    completed = 0
    episode_index = 0
    carry = env.reset()
    num_envs = env.num_envs
    command_chunks = torch.zeros(
        num_envs, HAIC_ACTION_HORIZON, HAIC_LATENT_DIM, device=env.device
    )
    rows_by_slot: list[list[dict[str, Any]]] = [[] for _ in range(num_envs)]
    actor_inputs: list[list[np.ndarray]] = [[] for _ in range(num_envs)]
    terminations: list[list[bool]] = [[] for _ in range(num_envs)]
    control_traces: list[list[dict[str, Any]]] = [[] for _ in range(num_envs)]
    raw_steps = 0

    def write_episode(slot: int) -> None:
        nonlocal accepted, episode_index
        rows = rows_by_slot[slot]
        if not rows or accepted >= target_episodes:
            return
        for row in rows:
            start = row["issued_raw_frame"]
            aligned = np.asarray(
                actor_inputs[slot][start : start + HAIC_ACTION_HORIZON],
                dtype=np.float32,
            )
            mask = np.asarray(
                terminations[slot][start : start + HAIC_ACTION_HORIZON],
                dtype=bool,
            )
            aligned = np.pad(
                aligned,
                ((0, HAIC_ACTION_HORIZON - len(aligned)), (0, 0)),
                mode="edge",
            )
            mask = np.pad(
                mask,
                (0, HAIC_ACTION_HORIZON - len(mask)),
                constant_values=True,
            )
            command = torch.as_tensor(
                row["action"], device=env.device, dtype=torch.float32
            )
            decoder_input = torch.cat(
                (torch.as_tensor(aligned, device=env.device), command), dim=-1
            )
            row["actor_input"] = aligned
            row["teacher_action"] = decoder(decoder_input).cpu().numpy()
            row["termination"] = mask
        shard = output_dir / f".bootstrap_{episode_index:06d}.mp4"
        _encode_video([row["rgb"] for row in rows], shard)
        arrays = {
            name: np.stack([row[name] for row in rows]).astype(np.float32)
            for name in ("state", "action", "actor_input", "teacher_action")
        }
        arrays["termination"] = np.stack(
            [row["termination"] for row in rows]
        ).astype(bool)
        arrays.update(
            {
                "control_applied_command": np.stack(
                    [record["applied_command"] for record in control_traces[slot]
                    ]
                ).astype(np.float32),
                "control_source_raw_frame": np.asarray(
                    [record["source_raw_frame"] for record in control_traces[slot]],
                    dtype=np.int64,
                ),
                "control_source_obs_id": np.asarray(
                    [record["source_obs_id"] for record in control_traces[slot]],
                    dtype=np.int64,
                ),
                "control_chunk_index": np.asarray(
                    [record["chunk_index"] for record in control_traces[slot]],
                    dtype=np.int64,
                ),
                "control_reward": np.asarray(
                    [record["reward"] for record in control_traces[slot]],
                    dtype=np.float32,
                ),
                "control_done": np.asarray(
                    [record["done"] for record in control_traces[slot]],
                    dtype=bool,
                ),
            }
        )
        arrays.update(
            {
                name: np.asarray([row[name] for row in rows])
                for name in (
                    "episode_id",
                    "obs_id",
                    "issued_raw_frame",
                    "ready_raw_frame",
                    "latency_ms",
                    "worker_id",
                )
            }
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

    while accepted < target_episodes and simulation_app.is_running():
        due_slots = [
            slot
            for slot, frame in enumerate(transport.frames)
            if frame % transport.clock.obs_stride_raw_frames == 0
        ]
        state = canonical_state(carry)
        state_cpu = state.detach().cpu().numpy().astype(np.float32, copy=False)
        if due_slots:
            command = teacher_command(policy, carry)
            command_chunks[:] = command.reshape(
                num_envs, HAIC_ACTION_HORIZON, HAIC_LATENT_DIM
            )
            rgb = refresh_rgb(env, due_slots, update_hz=HAIC_CONTROL_HZ)
            for index, slot in enumerate(due_slots):
                rows_by_slot[slot].append(
                    {
                        "rgb": rgb[index].copy(),
                        "state": state_cpu[slot].copy(),
                        "action": command_chunks[slot].cpu().numpy().copy(),
                    }
                )
        for slot in range(num_envs):
            actor_inputs[slot].append(state_cpu[slot].copy())

        action_td = carry.clone(False)
        action_td["action"] = command_chunks.reshape(num_envs, -1)
        td, carry = env.step_and_maybe_reset(action_td)
        raw_steps += num_envs
        for trace in env.base_env.latency_last_trace:
            for index, slot in enumerate(trace["env_ids"].detach().cpu().tolist()):
                application = trace["application"][index]
                control_traces[slot].append(
                    {
                        "applied_command": trace["applied_command"][slot]
                        .detach()
                        .cpu()
                        .numpy()
                        .copy(),
                        "source_raw_frame": application["source_raw_frame"],
                        "source_obs_id": application["source_obs_id"],
                        "chunk_index": application["chunk_index"],
                        "reward": trace["reward"][slot].sum().item(),
                        "done": bool(trace["done"][slot].item()),
                    }
                )
        for slot in due_slots:
            submission = env.base_env.latency_last_submission[slot]
            row = rows_by_slot[slot][-1]
            if submission is None:
                rows_by_slot[slot].pop()
            else:
                row.update(submission)
        done = td["next", "done"].squeeze(-1).bool()
        success = td["next", "stats", "success"].squeeze(-1).bool()
        for slot in range(num_envs):
            terminations[slot].append(bool(done[slot].item()))
        for slot in done.nonzero(as_tuple=False).flatten().cpu().tolist():
            completed += 1
            if (args.keep_failed or success[slot].item()) and accepted < target_episodes:
                write_episode(slot)
            rows_by_slot[slot] = []
            actor_inputs[slot] = []
            terminations[slot] = []
            control_traces[slot] = []
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
        "completed_episodes": completed,
        "raw_steps": raw_steps,
        "keep_failed": args.keep_failed,
        "issued_command_shape": [HAIC_ACTION_HORIZON, 256],
        "actor_input_shape": [HAIC_ACTION_HORIZON, 605],
        "teacher_action_shape": [HAIC_ACTION_HORIZON, 23],
        "termination_shape": [HAIC_ACTION_HORIZON],
    }
    _write_metadata(
        output_dir,
        {
            **result,
            "schema_version": 6,
            "shard_format": "npz+mp4",
            "shard_root": "rollout_shards",
            "rows_unit": "control_step",
            "env_fps": 50,
            "vla_fps": 10,
            "state_dim": 605,
            "vla_action_dim": 256,
            "issued_command_shape": [HAIC_ACTION_HORIZON, 256],
            "teacher_latent_dim": 256,
            "actor_input_shape": [HAIC_ACTION_HORIZON, 605],
            "teacher_action_shape": [HAIC_ACTION_HORIZON, 23],
            "termination_shape": [HAIC_ACTION_HORIZON],
            "prompt": "Pull the cart along the reference motion.",
        },
    )
    (output_dir / "bootstrap-collect.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return result


def _flush_dagger(
    output_dir: Path, shard_index: int, rows: list[dict[str, Any]], *, split: str = "train"
) -> None:
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
    if np.asarray(arrays["termination"]).ndim == 1:
        arrays["termination"][-1] = True
    arrays["image_shape"] = rows[0]["rgb"].shape
    write_haic_dagger_shard(
        output_dir,
        split=split,
        episode_idx=shard_index,
        arrays=arrays,
        video_path=video_path,
    )
    video_path.unlink()


@torch.inference_mode()
def _teacher_collect(args, env, policy, simulation_app) -> dict[str, Any]:
    """Collect H1 teacher episodes with fresh RGB and latent at every 50 Hz step."""
    from active_adaptation.vla.runtime import (
        HAIC_CONTROL_HZ,
        canonical_state,
        refresh_rgb,
        student_actor_from_policy,
        teacher_action,
        teacher_latent,
    )

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    actor = student_actor_from_policy(policy, env.device)
    carry = env.reset()
    rows_by_slot = [[] for _ in range(env.num_envs)]
    accepted = 0
    completed = 0
    row_count = 0
    while accepted < args.episode_budget and simulation_app.is_running():
        rgb = refresh_rgb(env, update_hz=HAIC_CONTROL_HZ)
        state = canonical_state(carry)
        latent = teacher_latent(policy, carry)
        labels = teacher_action(policy, carry)
        states = state.cpu().numpy()
        latents = latent.cpu().numpy()
        actions = labels.cpu().numpy()
        for slot in range(env.num_envs):
            rows_by_slot[slot].append({
                "rgb": rgb[slot].copy(),
                "state": states[slot].copy(),
                "action": latents[slot].copy(),
                "actor_input": states[slot].copy(),
                "teacher_action": actions[slot].copy(),
                "termination": False,
            })
        action_td = carry.clone(False)
        action_td["action"] = actor(torch.cat((state, latent), dim=-1))
        td, carry = env.step_and_maybe_reset(action_td)
        done = td["next", "done"].squeeze(-1)
        success = td["next", "stats", "success"].squeeze(-1).bool()
        for slot in done.nonzero(as_tuple=False).flatten().cpu().tolist():
            completed += 1
            rows = rows_by_slot[slot]
            rows[-1]["termination"] = True
            if accepted < args.episode_budget and (args.keep_failed or success[slot].item()):
                _flush_dagger(output_dir, accepted, rows, split=args.split)
                row_count += len(rows)
                accepted += 1
            rows_by_slot[slot] = []
    if accepted != args.episode_budget:
        raise RuntimeError(
            f"teacher collector stopped at {accepted}/{args.episode_budget} episodes"
        )
    result = {
        "mode": args.mode,
        "split": args.split,
        "episodes": accepted,
        "completed_episodes": completed,
        "rows": row_count,
        "env_fps": HAIC_CONTROL_HZ,
        "vla_fps": HAIC_CONTROL_HZ,
        "action_horizon": 1,
        "teacher_checkpoint": str(args.teacher_checkpoint),
        "seed": args.seed,
    }
    (output_dir / f"teacher-collect-{args.split}.json").write_text(
        json.dumps(result, indent=2) + "\n", encoding="utf-8"
    )
    return result


@torch.inference_mode()
def _dagger_collect(args, env, policy, simulation_app) -> dict[str, Any]:
    from latency_bench.core.latency_distribution import derive_seed

    from active_adaptation.vla.runtime import (
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
        env.num_envs, args.action_horizon, HAIC_LATENT_DIM, device=env.device
    )
    episode_counts = [0] * env.num_envs
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
                    [
                        derive_seed(
                            args.seed,
                            vector_index=slot,
                            episode_idx=episode_counts[slot],
                        )
                        for slot in due_slots
                    ],
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
                episode_counts[slot] += 1
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
            "vla_fps": HAIC_CONTROL_HZ / args.vla_cadence,
            "prompt": "Pull the cart along the reference motion.",
        },
    )
    (output_dir / "dagger-collect.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
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
    eval_config = None
    if args.mode == "latency-eval":
        eval_config = load_config(args.eval_config)
        args.teacher_checkpoint = Path(eval_config["env"]["runtime_checkpoint_path"])
        args.output_dir = Path(eval_config["logging"]["output_dir"])
        args.num_envs = eval_config["evaluation"]["eval_parallel_envs"]
        args.max_steps = eval_config["evaluation"]["eval_max_steps"]
        args.seed = eval_config["experiment"]["seed"]
        args.device = eval_config["env"]["simulator_device"]

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
        elif args.mode == "teacher-collect":
            result = _teacher_collect(args, env, policy, simulation_app)
        elif args.mode == "dagger-collect":
            result = _dagger_collect(args, env, policy, simulation_app)
        elif args.mode == "latency-eval":
            result = _run_latency_eval(eval_config, env, policy)
        else:
            result = _oracle_eval(args, env, policy, simulation_app)
        print(json.dumps(result, indent=2, sort_keys=True), flush=True)
    except BaseException:
        traceback.print_exc()
        exit_code = 1
    finally:
        env.close()
        if exit_code:
            # Kit's immediate shutdown exits with status 0 before Python resumes.
            sys.stdout.flush()
            sys.stderr.flush()
            os._exit(exit_code)
        simulation_app.close(skip_cleanup=True)


if __name__ == "__main__":
    main()
