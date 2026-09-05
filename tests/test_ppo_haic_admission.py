"""Native PPO regressions for admitted updates and finite long-chunk gradients.

Learning imports stay inside tests/workers so spawned ranks configure their
distributed environment before loading active_adaptation.
"""

import os
from pathlib import Path
from types import SimpleNamespace

import torch
import torch.distributed as dist
import torch.multiprocessing as mp


class _FiniteUpdateActor(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.log_scale = torch.nn.Parameter(torch.zeros(2))

    def get_dist(self, tensordict):
        loc = torch.zeros_like(tensordict["loc"])
        scale = self.log_scale.exp().expand_as(loc)
        from active_adaptation.learning.modules.distributions import IndependentNormal

        return IndependentNormal(loc, scale)


class _FiniteUpdateCritic(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.value = torch.nn.Parameter(torch.zeros(1))

    def forward(self, tensordict):
        values = self.value.expand(tensordict["ret"].shape)
        return {"state_value": values}


def test_ppo_update_stays_finite_for_extreme_log_ratio():
    from tensordict import TensorDict
    from active_adaptation.learning.modules.distributions import IndependentNormal
    from active_adaptation.learning.ppo.ppo_haic import PPOHAIC

    actor = _FiniteUpdateActor()
    critic = _FiniteUpdateCritic()
    trainer = SimpleNamespace(
        cfg=SimpleNamespace(
            phase="latency_command", normalize_ratio=False, max_grad_norm=1.0
        ),
        device=torch.device("cpu"),
        object_transform=torch.nn.Identity(),
        encoder_priv=torch.nn.Identity(),
        command_actor=actor,
        critic=critic,
        critic_loss_fn=torch.nn.MSELoss(reduction="none"),
        dist_keys=IndependentNormal.dist_keys,
        dist_cls=IndependentNormal,
        clip_param=0.2,
        entropy_coef=0.001,
        reward_groups=["reward"],
        opt_policy=torch.optim.Adam(actor.parameters(), lr=1e-3),
        opt_critic=torch.optim.Adam(critic.parameters(), lr=1e-3),
    )
    tensordict = TensorDict(
        {
            "loc": torch.zeros(2, 1, 2),
            "scale": torch.ones(2, 1, 2),
            "action": torch.zeros(2, 1, 2),
            "sample_log_prob": torch.full((2, 1), -1000.0),
            "adv": torch.tensor([[[1.0]], [[-1.0]]]),
            "step_count": torch.full((2, 1, 1), 6),
            "next": TensorDict(
                {"command_admitted": torch.ones(2, 1, 1, dtype=torch.bool)},
                batch_size=[2, 1],
            ),
            "ret": torch.tensor([[[1.0]], [[2.0]]]),
        },
        batch_size=[2, 1],
    )

    info = PPOHAIC._update_ppo(trainer, tensordict)

    assert all(torch.isfinite(value).all() for value in info.values())
    assert all(
        parameter.grad is not None and torch.isfinite(parameter.grad).all()
        for module in (actor, critic)
        for parameter in module.parameters()
    )


def _count_worker(rank: int, init_file: str, result_dir: str) -> None:
    os.environ["RANK"] = str(rank)
    os.environ["LOCAL_RANK"] = str(rank)
    os.environ["WORLD_SIZE"] = "2"
    dist.init_process_group(
        backend="gloo",
        init_method=f"file://{init_file}",
        rank=rank,
        world_size=2,
    )
    from active_adaptation.learning.ppo.ppo_haic import _global_count

    count = _global_count(torch.tensor([rank == 0], dtype=torch.bool))
    Path(result_dir, f"rank-{rank}").write_text(str(int(count.item())))
    dist.destroy_process_group()


def test_mixed_rank_admission_has_one_global_update(tmp_path):
    init_file = tmp_path / "process-group"
    result_dir = tmp_path / "results"
    result_dir.mkdir()
    mp.spawn(
        _count_worker,
        args=(str(init_file), str(result_dir)),
        nprocs=2,
        join=True,
    )
    assert [(result_dir / f"rank-{rank}").read_text() for rank in range(2)] == [
        "1",
        "1",
    ]
