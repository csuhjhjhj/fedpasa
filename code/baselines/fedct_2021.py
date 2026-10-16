"""FedCT (2021) in-domain reproduction."""

from .ops import MappingAdapter, semantic_weights
from .spec import BaselineSpec


Implementation = MappingAdapter
Aggregation = semantic_weights


SPEC = BaselineSpec(
    key="fedct",
    name="FedCT",
    year=2021,
    paper_role="collaborative transfer with semantic aggregation and proximal regularization",
    args=(
        "--method", "transfr",
        "--server_optimizer", "fedavg",
        "--aggregation", "semantic",
        "--semantic_temperature", "0.1",
        "--semantic_self_weight", "0.2",
        "--semantic_anchor_weight", "0",
        "--prox_mu", "0.1",
    ),
)
