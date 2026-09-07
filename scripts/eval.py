import torch
import hydra
import numpy as np
import einops
import time
import sys
from tqdm import tqdm
from omegaconf import OmegaConf, DictConfig

from isaaclab.app import AppLauncher

import wandb
import logging
from tqdm import tqdm
from helpers import make_env_policy, evaluate
from active_adaptation.vla.runtime import teacher_latent_rollout

import os
import datetime
import termcolor
from pathlib import Path
from latency_bench.core.config import load_config
from hydra.core.hydra_config import HydraConfig

@hydra.main(config_path="../cfg", config_name="eval", version_base=None)
def main(cfg: DictConfig):
    OmegaConf.resolve(cfg)
    OmegaConf.set_struct(cfg, False)
    
    app_launcher = AppLauncher(OmegaConf.to_container(cfg.app))
    simulation_app = app_launcher.app
    output_dir = Path(HydraConfig.get().runtime.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    env, agent, vecnorm = make_env_policy(cfg, restore_latency_state=False)
    
    keys = [
        ("next", "stats"),
        ("next", "done"), 
        ("next", "reward"),
        "value_obs",
        "value_priv",
        "value_adapt",
        "context_expert",
        "context_scale",
        "context_adapt",
        "context_adapt_scale",
        "action_kl",
    ]

    policy_keys = ["dr_", "dr_pred"]
    
    eval_teacher = cfg.task.latency_command and cfg.algo.eval_teacher
    if eval_teacher:
        policy_eval = lambda tensordict: teacher_latent_rollout(agent, tensordict)
    else:
        policy_eval = agent.get_rollout_policy("eval")
    render_mode = cfg.get("render_mode", "rgb_array")
    info, trajs, stats, policy_trajs = evaluate(env, policy_eval, render=cfg.eval_render, render_mode=render_mode, seed=cfg.seed, keys=keys, policy_keys=policy_keys)
    
    # print(termcolor.colored(trajs, "light_yellow"))
    # time_str = datetime.datetime.now().strftime("%m-%d_%H-%M")
    # path = os.path.join(os.path.dirname(__file__), f"trajs-{time_str}.pt")
    # torch.save(trajs, path)
    
    torch.save(policy_trajs, output_dir / "policy_trajs.pt")

    # path = os.path.join(os.path.dirname(__file__), f"stats-{time_str}.pt")
    # torch.save(stats, path)

    info["task"] = cfg.task.name
    info["algo"] = cfg.algo.name
    info["checkpoint_path"] = cfg.checkpoint_path
    info["teacher_baseline"] = eval_teacher
    info["env_fps"] = 1.0 / env.base_env.step_dt
    if cfg.task.latency_command:
        latency_config = load_config(cfg.task.latency_config_path)
        info["latency_method"] = latency_config["latency"]["method"]
        info["latency_config_path"] = cfg.task.latency_config_path
        if info["latency_method"] == "temporal":
            info["latency_profile_path"] = latency_config["latency"]["profile_path"]
        else:
            info["latency_profile_path"] = None
        info["obs_fps"] = latency_config["env"]["obs_fps"]
    else:
        info["latency_method"] = None
        info["latency_config_path"] = None
        info["latency_profile_path"] = None
        info["obs_fps"] = None
    info["argv"] = sys.argv
    print(OmegaConf.to_yaml(info))
    
    time_str = datetime.datetime.now().strftime("%m-%d_%H-%M")
    path = output_dir / f"{cfg.task.name}-{time_str}.yaml"
    with open(path, "w") as f:
        OmegaConf.save(info, f)
    print(termcolor.colored(f"Saved results to: {path}", "green"))

    os._exit(0)
    env.close()
    simulation_app.close()


if __name__ == "__main__":
    main()
