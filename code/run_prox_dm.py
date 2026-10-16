"""Luxury fix: prox sweep on all in-domain datasets + digital_music in-domain matrix."""
import sys
from pathlib import Path
sys.path.insert(0, '.')
import run_www_matrix as M

# 1) fedpasa + prox_mu {0.05, 0.1} on 4 in-domain datasets
for ds, cfg in M.INDOMAIN_DS.items():
    for pm in ("0.05", "0.1"):
        for seed in M.SEEDS:
            run_dir = M.OUT / "prox" / f"in_{ds}_pm{pm}" / "fedpasa" / f"seed{seed}"
            cmd = M.COMMON + M.ds_args(ds, cfg) + list(M.FEDPASA) + [
                "--prox_mu", pm, "--output_dir", str(run_dir), "--seed", str(seed)]
            M.run_one(cmd, run_dir, dict(stage="prox", group=f"pm{pm}", method="fedpasa",
                                         dataset=ds, source="", target=ds, seed=seed))

# 2) digital_music as an in-domain dataset for all 11 methods
DM = dict(dat=Path("data/amazon_v2/digital_music/digital_music.dat"),
          titles=Path("data/amazon_v2/digital_music/titles.json"),
          clients=10, rounds=20, steps=40, eval_users=0)
for method, (needs_init, margs) in M.INDOMAIN_METHODS.items():
    for seed in M.SEEDS:
        run_dir = M.OUT / "indomain" / "digital_music" / method / f"seed{seed}"
        cmd = M.COMMON + M.ds_args("digital_music", DM) + margs + [
            "--output_dir", str(run_dir), "--seed", str(seed)]
        M.run_one(cmd, run_dir, dict(stage="indomain", group="indomain", method=method,
                                     dataset="digital_music", source="", target="digital_music", seed=seed))
print("[prox+dm] ALL DONE")
