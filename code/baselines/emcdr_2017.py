"""EMCDR (2017) in-domain reproduction.

EMCDR learns an explicit mapping between source and target representation spaces.
For in-domain experiments, this is represented by a text-to-collaborative adapter
trained with semantic alignment loss to preserve cross-domain structure, plus
zero-initialized local heads to simulate the learned mapping function.
"""

from .ops import MappingAdapter
from .spec import BaselineSpec


Implementation = MappingAdapter


SPEC = BaselineSpec(
    key="emcdr",
    name="EMCDR",
    year=2017,
    paper_role="mapping-based cross-domain recommendation with semantic alignment",
    args=(
        "--method", "transfr",
        "--server_optimizer", "fedavg",
        "--aggregation", "fedavg",
        "--semantic_anchor_weight", "0",
        "--prox_mu", "0",
        "--head_zero_init",
        "--align_weight", "0.1",
    ),
)
