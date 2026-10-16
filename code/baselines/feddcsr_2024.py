"""FedDCSR (2024) in-domain reproduction.

FedDCSR emphasizes disentangled cross-domain sequential recommendation. The
current in-domain reproduction uses dual adapters and decoupled update
aggregation to separate personalization and transferable update directions.
"""

from .ops import DisentangledDualAdapter, decoupled_direction_magnitude_weights
from .spec import BaselineSpec


Implementation = DisentangledDualAdapter
Aggregation = decoupled_direction_magnitude_weights


SPEC = BaselineSpec(
    key="feddcsr",
    name="FedDCSR",
    year=2024,
    paper_role="disentangled sequential recommendation via dual adapters and decoupled aggregation",
    args=(
        "--method", "dual",
        "--dual_personal_mix", "0.2",
        "--dual_personal_aggregation", "local",
        "--server_optimizer", "fedavg",
        "--aggregation", "decoupled",
        "--semantic_temperature", "0.1",
        "--semantic_self_weight", "0.2",
        "--semantic_anchor_weight", "0.2",
        "--prox_mu", "0",
    ),
)
