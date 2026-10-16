"""FedPer (NeurIPS 2020) lightweight matched-protocol reproduction."""

from .ops import DisentangledDualAdapter
from .spec import BaselineSpec


Implementation = DisentangledDualAdapter


SPEC = BaselineSpec(
    key="fedper",
    name="FedPer",
    year=2020,
    paper_role="shared adapter branch with local personalization branch",
    args=(
        "--method", "dual",
        "--server_optimizer", "fedavg",
        "--aggregation", "fedavg",
        "--dual_personal_mix", "0.5",
        "--dual_personal_aggregation", "local",
        "--semantic_anchor_weight", "0",
        "--prox_mu", "0",
    ),
)
