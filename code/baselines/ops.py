"""Reference implementation sketches for named baseline reproductions.

NOTE: These classes are **reference/documentation only** and are NOT imported
or used by ``run_transfr_style.py``. The actual executable implementations live
in ``run_transfr_style.py`` (classes ``Adapter``, ``FedBATAdapter``, ``DualAdapter``,
and functions ``aggregate_states``, ``personalized_aggregate_states``, etc.).

The ``Implementation`` and ``Aggregation`` exports in each ``baselines/*.py``
module are also unused at runtime — only ``SPEC.args`` is consumed by runner
scripts to build command lines for ``run_transfr_style.py``.

Keep this file as a human-readable sketch of the architectural intent behind
each baseline mapping. Do NOT rely on it for execution.
"""

from __future__ import annotations

from typing import Iterable

import torch
import torch.nn as nn
import torch.nn.functional as F


class MatrixFactorizationBaseline(nn.Module):
    """User-item matrix factorization used by CMF/FCF in-domain baselines."""

    def __init__(self, num_users: int, num_items: int, dim: int = 64):
        super().__init__()
        self.user = nn.Embedding(num_users, dim)
        self.item = nn.Embedding(num_items, dim)
        nn.init.normal_(self.user.weight, std=0.02)
        nn.init.normal_(self.item.weight, std=0.02)

    def forward(self, users: torch.Tensor, items: torch.Tensor) -> torch.Tensor:
        return (self.user(users) * self.item(items)).sum(dim=-1)


class MappingAdapter(nn.Module):
    """MLP mapping adapter used for EMCDR-style representation transfer."""

    def __init__(self, dim: int = 768, bottleneck: int = 64, layers: int = 2, dropout: float = 0.1):
        super().__init__()
        blocks = []
        for _ in range(max(1, layers)):
            blocks.append(
                nn.Sequential(
                    nn.LayerNorm(dim),
                    nn.Linear(dim, bottleneck),
                    nn.ReLU(),
                    nn.Dropout(dropout),
                    nn.Linear(bottleneck, dim),
                )
            )
        self.blocks = nn.ModuleList(blocks)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for block in self.blocks:
            x = x + block(x)
        return F.normalize(x, dim=-1)


class DisentangledDualAdapter(nn.Module):
    """Global/personal dual-branch adapter used by DisenCDR/FedDCSR."""

    def __init__(self, dim: int = 768, bottleneck: int = 64, personal_mix: float = 0.2):
        super().__init__()
        self.personal_mix = personal_mix
        self.global_branch = MappingAdapter(dim, bottleneck)
        self.personal_branch = MappingAdapter(dim, bottleneck)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        global_z = self.global_branch(x)
        personal_z = self.personal_branch(x)
        return F.normalize((1 - self.personal_mix) * global_z + self.personal_mix * personal_z, dim=-1)


def semantic_weights(profiles: torch.Tensor, temperature: float = 0.1, self_weight: float = 0.2) -> torch.Tensor:
    """Client-profile similarity weights used by IFedRec/FedCT-style baselines."""

    sim = profiles @ profiles.t()
    weights = torch.softmax(sim / max(temperature, 1e-6), dim=1)
    if self_weight > 0:
        eye = torch.eye(weights.size(0), device=weights.device, dtype=weights.dtype)
        weights = (1 - self_weight) * weights + self_weight * eye
        weights = weights / weights.sum(dim=1, keepdim=True).clamp_min(1e-12)
    return weights


def adaptive_weights(
    profiles: torch.Tensor,
    update_directions: torch.Tensor,
    temperature: float = 0.1,
    self_weight: float = 0.2,
    update_weight: float = 0.5,
) -> torch.Tensor:
    """Profile + update-direction weights used by the FELLRec-style baseline."""

    profile_sim = profiles @ profiles.t()
    update_sim = update_directions @ update_directions.t()
    sim = (1 - update_weight) * profile_sim + update_weight * update_sim
    weights = torch.softmax(sim / max(temperature, 1e-6), dim=1)
    if self_weight > 0:
        eye = torch.eye(weights.size(0), device=weights.device, dtype=weights.dtype)
        weights = (1 - self_weight) * weights + self_weight * eye
        weights = weights / weights.sum(dim=1, keepdim=True).clamp_min(1e-12)
    return weights


def decoupled_direction_magnitude_weights(
    profiles: torch.Tensor,
    update_vectors: Iterable[torch.Tensor],
    temperature: float = 0.1,
    self_weight: float = 0.2,
    profile_weight: float = 0.2,
    magnitude_weight: float = 0.2,
) -> torch.Tensor:
    """Direction/magnitude disentanglement used by the FedDCSR-style baseline."""

    deltas = [delta.detach().flatten().cpu() for delta in update_vectors]
    norms = torch.stack([delta.norm().clamp_min(1e-12) for delta in deltas])
    directions = torch.stack([delta / norm for delta, norm in zip(deltas, norms)])
    direction_sim = directions @ directions.t()
    profile_sim = profiles.cpu() @ profiles.cpu().t()
    log_norms = torch.log(norms.clamp_min(1e-12))
    magnitude_compat = torch.exp(-torch.abs(log_norms[:, None] - log_norms[None, :]))
    sim = (1 - profile_weight) * direction_sim + profile_weight * profile_sim
    sim = sim + magnitude_weight * magnitude_compat
    weights = torch.softmax(sim / max(temperature, 1e-6), dim=1)
    if self_weight > 0:
        eye = torch.eye(weights.size(0), dtype=weights.dtype)
        weights = (1 - self_weight) * weights + self_weight * eye
        weights = weights / weights.sum(dim=1, keepdim=True).clamp_min(1e-12)
    return weights
