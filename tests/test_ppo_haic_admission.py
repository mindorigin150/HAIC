"""Distributed admission is a global PPO update contract."""

import os
from pathlib import Path

import torch
import torch.distributed as dist
import torch.multiprocessing as mp


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
