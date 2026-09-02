"""Pure-PyTorch HAIC student actor.

This module deliberately has no TensorDict, TorchRL, Isaac Lab, or HAIC
runtime imports.  It is used by the GR00T process as the small trainable
controller after the VLA latent, and by :mod:`ppo_haic` through the aliases
below to keep the native policy implementation unchanged.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Mapping

import torch
from torch import nn


HAIC_STATE_DIM = 605
HAIC_PRIV_FEATURE_DIM = 256
HAIC_ACTION_DIM = 23
HAIC_ACTOR_INPUT_DIM = HAIC_STATE_DIM + HAIC_PRIV_FEATURE_DIM


def make_mlp(
    num_units: list[int] | tuple[int, ...],
    activation: type[nn.Module] = nn.Mish,
    norm: str | None = "before",
    dropout: float = 0.0,
    input_dim: int | None = None,
) -> nn.Sequential:
    """Build the native HAIC MLP with the original module ordering."""
    assert norm in ("before", "after", None)
    layers: list[nn.Module] = []
    lazy = input_dim is None
    for num_units_i in num_units:
        layers.append(
            nn.LazyLinear(num_units_i)
            if lazy
            else nn.Linear(input_dim, num_units_i)
        )
        input_dim = num_units_i
        if norm == "before":
            layers.extend((nn.LayerNorm(num_units_i), activation()))
        elif norm == "after":
            layers.extend((activation(), nn.LayerNorm(num_units_i)))
        else:
            layers.append(activation())
        if dropout > 0.0:
            layers.append(nn.Dropout(dropout))
    return nn.Sequential(*layers)


class Actor(nn.Module):
    """Native HAIC Gaussian head, kept byte-for-byte compatible in behavior."""

    def __init__(
        self,
        action_dim: int,
        init_noise_scale: float = 1.0,
        predict_std: bool = False,
        load_noise_scale: float | None = None,
        input_dim: int | None = None,
    ) -> None:
        super().__init__()
        self.predict_std = predict_std
        output_dim = action_dim * 2 if predict_std else action_dim
        self.actor_mean = (
            nn.LazyLinear(output_dim)
            if input_dim is None
            else nn.Linear(input_dim, output_dim)
        )
        if not predict_std:
            self.actor_std = nn.Parameter(torch.ones(action_dim) * init_noise_scale)
        self.scale_mapping = nn.Identity()
        self.load_noise_scale = load_noise_scale

    def forward(self, features: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if self.predict_std:
            loc, scale = self.actor_mean(features).chunk(2, dim=-1)
        else:
            loc = self.actor_mean(features)
            scale = torch.ones_like(loc) * self.actor_std
        return loc, self.scale_mapping(scale)

    def _load_from_state_dict(
        self,
        state_dict,
        prefix,
        local_metadata,
        strict,
        missing_keys,
        unexpected_keys,
        error_msgs,
    ):
        super()._load_from_state_dict(
            state_dict,
            prefix,
            local_metadata,
            strict,
            missing_keys,
            unexpected_keys,
            error_msgs,
        )
        if self.load_noise_scale is not None:
            self.actor_std.data.fill_(self.load_noise_scale)


class HaicStudentActor(nn.Module):
    """The deployable 605D state + 256D latent -> 23D HAIC actor."""

    def __init__(self) -> None:
        super().__init__()
        self.feature = make_mlp([512, 256, 256], input_dim=HAIC_ACTOR_INPUT_DIM)
        self.head = Actor(HAIC_ACTION_DIM, input_dim=HAIC_PRIV_FEATURE_DIM)

    def forward(self, actor_input: torch.Tensor) -> torch.Tensor:
        loc, _ = self.head(self.feature(actor_input))
        return loc


_ACTOR_ADAPT_KEYS = {
    "feature.0.weight": "module.0.module.1.module.0.weight",
    "feature.0.bias": "module.0.module.1.module.0.bias",
    "feature.1.weight": "module.0.module.1.module.1.weight",
    "feature.1.bias": "module.0.module.1.module.1.bias",
    "feature.3.weight": "module.0.module.1.module.3.weight",
    "feature.3.bias": "module.0.module.1.module.3.bias",
    "feature.4.weight": "module.0.module.1.module.4.weight",
    "feature.4.bias": "module.0.module.1.module.4.bias",
    "feature.6.weight": "module.0.module.1.module.6.weight",
    "feature.6.bias": "module.0.module.1.module.6.bias",
    "feature.7.weight": "module.0.module.1.module.7.weight",
    "feature.7.bias": "module.0.module.1.module.7.bias",
    "head.actor_mean.weight": "module.0.module.2.module.actor_mean.weight",
    "head.actor_mean.bias": "module.0.module.2.module.actor_mean.bias",
    "head.actor_std": "module.0.module.2.module.actor_std",
}


def extract_actor_adapt_state_dict(
    policy_state: Mapping[str, torch.Tensor],
) -> dict[str, torch.Tensor]:
    """Extract the native ``policy.actor_adapt`` weights by exact key.

    The explicit map is intentional: a checkpoint with a different actor
    topology must fail at the missing key instead of silently selecting a
    similarly named module.
    """
    return {
        target_key: policy_state[source_key]
        for target_key, source_key in _ACTOR_ADAPT_KEYS.items()
    }


def load_actor_adapt(
    checkpoint_path: str | Path,
    *,
    device: torch.device | str = "cpu",
) -> HaicStudentActor:
    """Load ``policy.actor_adapt`` from a native HAIC checkpoint."""
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    actor = HaicStudentActor().to(device)
    actor.load_state_dict(
        extract_actor_adapt_state_dict(checkpoint["policy"]["actor_adapt"])
    )
    actor.eval()
    return actor


def main() -> None:
    parser = argparse.ArgumentParser(description="Export the HAIC actor_adapt state")
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    actor = load_actor_adapt(args.checkpoint)
    temporary = args.output.with_suffix(".pt.tmp")
    torch.save(actor.state_dict(), temporary)
    temporary.replace(args.output)


if __name__ == "__main__":
    main()
