"""FELLRec (2025) in-domain reproduction.

The official FELLRec code path is handled separately for RecFormer runs. This
in-domain lightweight reproduction captures its personalized aggregation role
with adaptive aggregation plus a Ditto-style local personalization stage.
"""

from .ops import MappingAdapter, adaptive_weights
from .spec import BaselineSpec


Implementation = MappingAdapter
Aggregation = adaptive_weights


SPEC = BaselineSpec(
    key="fellrec",
    name="FELLRec",
    year=2025,
    paper_role="adaptive personalized aggregation with local post-adaptation",
    args=(
        "--method", "transfr",
        "--server_optimizer", "fedavg",
        "--aggregation", "adaptive",
        "--semantic_temperature", "0.1",
        "--semantic_self_weight", "0.2",
        "--ditto_steps", "40",
        "--ditto_mu", "0.1",
        "--semantic_anchor_weight", "0",
        "--prox_mu", "0",
    ),
)
