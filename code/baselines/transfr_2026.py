"""TransFR (2026) in-domain reproduction."""

from .ops import MappingAdapter
from .spec import BaselineSpec


Implementation = MappingAdapter


SPEC = BaselineSpec(
    key="transfr",
    name="TransFR",
    year=2026,
    paper_role="federated text adapter with FedAvg aggregation",
    args=(
        "--method", "transfr",
        "--server_optimizer", "fedavg",
        "--aggregation", "fedavg",
        "--semantic_anchor_weight", "0",
        "--prox_mu", "0",
    ),
)
