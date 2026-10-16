"""CMF (2008) in-domain reproduction.

CMF (Collective Matrix Factorization) jointly factorizes user-item and item-item
relation matrices. In the current in-domain setting there is no separate source
matrix, so we approximate the collective aspect by training a federated MF with
higher latent dimensionality and more communication rounds, reflecting CMF's
richer joint factorization space compared to plain federated CF.
"""

from .ops import MatrixFactorizationBaseline
from .spec import BaselineSpec


Implementation = MatrixFactorizationBaseline


SPEC = BaselineSpec(
    key="cmf",
    name="CMF",
    year=2008,
    paper_role="collective matrix factorization approximated by higher-dimensional federated MF",
    args=(
        "--only_mf_baseline",
        "--mf_dim", "128",
        "--mf_rounds", "30",
        "--mf_local_steps", "100",
        "--mf_lr", "5e-4",
    ),
)
