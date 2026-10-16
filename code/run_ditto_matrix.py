"""FedPASA + Moreau (ditto) personalization: cross 6x5 + indomain 4x5."""
import sys
from pathlib import Path
sys.path.insert(0, '.')
import run_www_matrix as M

ARGS = list(M.FEDPASA) + ["--ditto_steps", "40", "--ditto_mu", "0.15"]
for source, target in M.DIRECTIONS:
    for seed in M.SEEDS:
        run_dir = M.OUT / "ditto" / f"{source}_to_{target}" / "pasa_ditto" / f"seed{seed}"
        cmd = M.COMMON + M.ds_args(target, M.CROSS_DS[target]) + ARGS + [
            "--output_dir", str(run_dir), "--seed", str(seed),
            "--init_adapter_path", str(M.OUT / "pretrain" / f"source_{source}_seed{seed}_adapter.pt")]
        M.run_one(cmd, run_dir, dict(stage="ditto", group="cross", method="pasa_ditto",
                                     dataset=target, source=source, target=target, seed=seed))
for ds, cfg in M.INDOMAIN_DS.items():
    for seed in M.SEEDS:
        run_dir = M.OUT / "ditto" / f"in_{ds}" / "pasa_ditto" / f"seed{seed}"
        cmd = M.COMMON + M.ds_args(ds, cfg) + ARGS + [
            "--output_dir", str(run_dir), "--seed", str(seed)]
        M.run_one(cmd, run_dir, dict(stage="ditto", group=f"in_{ds}", method="pasa_ditto",
                                     dataset=ds, source="", target=ds, seed=seed))
print("[ditto] ALL DONE")
