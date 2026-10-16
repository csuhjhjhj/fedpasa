"""Enhancement variants matrix: learnable per-client lambda and cold-weighted anchor.

Variants:
  pasa_ll   = FedPASA + --fedbat_learn_lambda 1
  pasa_cd   = FedPASA + --semantic_anchor_cold_delta 0.1
  pasa_both = FedPASA + both
Runs cross (6 dirs x 5 seeds) and indomain (4 ds x 5 seeds). Reuses the
same source adapters and protocol as the main matrix.
"""
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
import run_www_matrix as M

VARIANTS = {
    "pasa_ll":   M.FEDPASA + ["--fedbat_learn_lambda", "1"],
    "pasa_cd":   M.FEDPASA + ["--semantic_anchor_cold_delta", "0.1"],
    "pasa_both": M.FEDPASA + ["--fedbat_learn_lambda", "1", "--semantic_anchor_cold_delta", "0.1"],
}


def main():
    for vname, margs in VARIANTS.items():
        # cross
        for source, target in M.DIRECTIONS:
            for seed in M.SEEDS:
                run_dir = M.OUT / "enhance" / f"{source}_to_{target}" / vname / f"seed{seed}"
                cmd = M.COMMON + M.ds_args(target, M.CROSS_DS[target]) + margs + [
                    "--output_dir", str(run_dir), "--seed", str(seed),
                    "--init_adapter_path", str(M.OUT / "pretrain" / f"source_{source}_seed{seed}_adapter.pt")]
                M.run_one(cmd, run_dir, dict(stage="enhance", group=vname, method=vname,
                                             dataset=target, source=source, target=target, seed=seed))
        # indomain
        for ds, cfg in M.INDOMAIN_DS.items():
            for seed in M.SEEDS:
                run_dir = M.OUT / "enhance" / f"in_{ds}" / vname / f"seed{seed}"
                cmd = M.COMMON + M.ds_args(ds, cfg) + margs + [
                    "--output_dir", str(run_dir), "--seed", str(seed)]
                M.run_one(cmd, run_dir, dict(stage="enhance", group=f"in_{ds}", method=vname,
                                             dataset=ds, source="", target=ds, seed=seed))
    print("[enhance] ALL DONE", flush=True)


if __name__ == "__main__":
    main()
