"""FCF (2019) in-domain reproduction.

FCF (Federated Collaborative Filtering) focuses on communication-efficient
federated matrix factorization with fewer rounds but more local computation.
We match this design with moderate dimensionality, fewer communication rounds,
and larger local step counts relative to CMF.
"""

from .ops import MatrixFactorizationBaseline
from .spec import BaselineSpec


Implementation = MatrixFactorizationBaseline


SPEC = BaselineSpec(
    key="fcf",
    name="FCF",
    year=2019,
    paper_role="federated collaborative filtering with communication-efficient design",
    args=(
        "--only_mf_baseline",
        "--mf_dim", "64",
        "--mf_rounds", "25",
        "--mf_local_steps", "80",
        "--mf_lr", "1e-3",
    ),
)
