"""pFedMe (ICML 2020) lightweight matched-protocol reproduction."""

from .ops import MappingAdapter
from .spec import BaselineSpec


Implementation = MappingAdapter


SPEC = BaselineSpec(
    key="pfedme",
    name="pFedMe",
    year=2020,
    paper_role="parameter-space personalization with Moreau-style proximal local adaptation",
    args=(
        "--method", "transfr",
        "--server_optimizer", "fedavg",
        "--aggregation", "fedavg",
        "--prox_mu", "0.15",
        "--ditto_steps", "40",
        "--ditto_mu", "0.15",
        "--semantic_anchor_weight", "0",
    ),
)
