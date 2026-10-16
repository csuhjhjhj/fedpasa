import subprocess, sys
from pathlib import Path
sys.path.insert(0, '.')
import run_www_matrix as M
for source, target in M.DIRECTIONS:
    for seed in M.SEEDS:
        run_dir = M.OUT / "nosrc" / f"{source}_to_{target}" / "pasa_nosrc" / f"seed{seed}"
        cmd = M.COMMON + M.ds_args(target, M.CROSS_DS[target]) + list(M.FEDPASA) + [
            "--output_dir", str(run_dir), "--seed", str(seed)]
        M.run_one(cmd, run_dir, dict(stage="nosrc", group="pasa_nosrc", method="pasa_nosrc",
                                     dataset=target, source=source, target=target, seed=seed))
print("[nosrc] DONE")
