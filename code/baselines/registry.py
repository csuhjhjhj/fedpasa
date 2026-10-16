"""Registry for named baseline reproductions."""

from __future__ import annotations

from .cmf_2008 import SPEC as CMF
from .disencdr_2022 import SPEC as DISENCDR
from .emcdr_2017 import SPEC as EMCDR
from .fcf_2019 import SPEC as FCF
from .fedct_2021 import SPEC as FEDCT
from .feddcsr_2024 import SPEC as FEDDCSR
from .fellrec_2025 import SPEC as FELLREC
from .fedper_2020 import SPEC as FEDPER
from .ifedrec_2024 import SPEC as IFEDREC
from .pfedme_2020 import SPEC as PFEDME
from .transfr_2026 import SPEC as TRANSFR
from .transfr_fixedbranch_2026 import SPEC as FIXEDBRANCH
from .transfr_l2sp_2018 import SPEC as L2SP
from .spec import BaselineSpec


BASELINES = {
    spec.key: spec
    for spec in [
        CMF,
        EMCDR,
        DISENCDR,
        FCF,
        IFEDREC,
        PFEDME,
        FEDPER,
        FELLREC,
        FEDCT,
        FEDDCSR,
        TRANSFR,
        FIXEDBRANCH,
        L2SP,
    ]
}
