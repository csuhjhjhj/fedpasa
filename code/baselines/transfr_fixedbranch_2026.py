"""TransFR + FixedBranch ablation baseline."""

from .ops import DisentangledDualAdapter
from .spec import BaselineSpec


Implementation = DisentangledDualAdapter


SPEC = BaselineSpec(
    key="fixedbranch",
    name="TransFR+FixedBranch",
    year=2026,
    paper_role="TransFR with warm/cold branching only, no client conditioning or anchoring",
    args=(
        "--method", "fedbat",
        "--server_optimizer", "fedavg",
        "--aggregation", "fedavg",
        "--fedbat_use_cold", "1",
        "--fedbat_alpha", "1.0",
        "--fedbat_condition_scale", "0.0",
        "--semantic_anchor_weight", "0",
        "--semantic_anchor_auto", "0",
        "--prox_mu", "0",
    ),
)
