"""IFedRec (2024) in-domain reproduction.

IFedRec is represented as text-enhanced federated recommendation with
client-profile-aware semantic aggregation.
"""

from .ops import MappingAdapter, semantic_weights
from .spec import BaselineSpec


Implementation = MappingAdapter
Aggregation = semantic_weights


SPEC = BaselineSpec(
    key="ifedrec",
    name="IFedRec",
    year=2024,
    paper_role="text-enhanced federated recommendation with semantic personalization",
    args=(
        "--method", "transfr",
        "--server_optimizer", "fedavg",
        "--aggregation", "semantic",
        "--semantic_temperature", "0.1",
        "--semantic_self_weight", "0.2",
        "--profile_score_weight", "0.05",
        "--semantic_anchor_weight", "0",
        "--prox_mu", "0",
    ),
)
