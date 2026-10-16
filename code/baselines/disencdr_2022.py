"""DisenCDR (2022) in-domain reproduction.

DisenCDR separates transferable and domain-specific factors. The in-domain
adapter reproduction uses a dual global/personal adapter with a light semantic
anchor.
"""

from .ops import DisentangledDualAdapter
from .spec import BaselineSpec


Implementation = DisentangledDualAdapter


SPEC = BaselineSpec(
    key="disencdr",
    name="DisenCDR",
    year=2022,
    paper_role="disentangled transfer represented by dual global/personal adapters",
    args=(
        "--method", "dual",
        "--dual_personal_mix", "0.2",
        "--dual_personal_aggregation", "local",
        "--server_optimizer", "fedavg",
        "--aggregation", "fedavg",
        "--semantic_anchor_weight", "0.2",
        "--prox_mu", "0",
    ),
)
