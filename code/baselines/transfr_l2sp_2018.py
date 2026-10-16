"""TransFR + L2-SP transfer regularization baseline."""

from .ops import MappingAdapter
from .spec import BaselineSpec


Implementation = MappingAdapter


SPEC = BaselineSpec(
    key="l2sp",
    name="TransFR+L2SP",
    year=2018,
    paper_role="TransFR with fixed parameter-level L2-SP anchoring to source adapter",
    args=(
        "--method", "transfr",
        "--server_optimizer", "fedavg",
        "--aggregation", "fedavg",
        "--l2sp_mu", "0.01",
        "--semantic_anchor_weight", "0",
        "--prox_mu", "0",
    ),
)
