"""Gap-closure experiments: validation-based tuning, fixed-compute scaling, partial participation."""
import sys
from pathlib import Path
sys.path.insert(0, '.')
import run_www_matrix as M

SRC, TGT = "music", "movies"
SEEDS = [42, 123, 456]
ADAPTER = M.OUT / "pretrain" / f"source_{SRC}_seed{{}}_adapter.pt"

BASE = [a for a in M.COMMON if a != "--skip_valid_eval"]  # keep validation eval for the tuning sweep

def cmd_cross(margs, seed, extra=None, skip_valid=True):
    cfg = M.CROSS_DS[TGT]
    common = M.COMMON if skip_valid else BASE
    cmd = common + M.ds_args(TGT, cfg) + margs + [
        "--output_dir", None, "--seed", str(seed),
        "--init_adapter_path", str(M.OUT / "pretrain" / f"source_{SRC}_seed{seed}_adapter.pt")]
    if extra:
        cmd += extra
    return cmd

# ---- 1. validation-based hyperparameter sweep (base framework, valid split) ----
print("== validation sweep", flush=True)
for alpha in ("0.4", "0.55", "0.7", "0.85"):
    for kappa in ("0.5", "0.7", "1.0"):
        for seed in SEEDS:
            run_dir = M.OUT / "validsweep" / f"a{alpha}_k{kappa}" / f"seed{seed}"
            margs = list(M.FEDPASA) + ["--fedbat_alpha", alpha, "--fedbat_cold_power", kappa]
            cmd = cmd_cross(margs, seed, skip_valid=False)
            cmd[cmd.index(None)] = str(run_dir)
            M.run_one(cmd, run_dir, dict(stage="validsweep", group=f"a{alpha}_k{kappa}", method="fedpasa",
                                         dataset=TGT, source=SRC, target=TGT, seed=seed))

# ---- 2. fixed-compute client scaling ----
print("== fixed-compute scaling", flush=True)
for nclients, steps in ((5, 30), (20, 8), (50, 3)):
    for method in ("transfr", "fedpasa"):
        for seed in SEEDS:
            run_dir = M.OUT / "fixedcompute" / f"clients{nclients}" / method / f"seed{seed}"
            margs = M.CROSS_METHODS[method][1]
            cfg = dict(M.CROSS_DS[TGT]); cfg["clients"] = nclients; cfg["steps"] = steps
            cmd = M.COMMON + M.ds_args(TGT, cfg) + margs + [
                "--output_dir", str(run_dir), "--seed", str(seed),
                "--init_adapter_path", str(M.OUT / "pretrain" / f"source_{SRC}_seed{seed}_adapter.pt")]
            M.run_one(cmd, run_dir, dict(stage="fixedcompute", group=f"clients{nclients}", method=method,
                                         dataset=TGT, source=SRC, target=TGT, seed=seed))

# ---- 3. partial participation ----
print("== partial participation", flush=True)
for rate in ("0.2", "0.5"):
    for method in ("transfr", "fedpasa"):
        for seed in SEEDS:
            run_dir = M.OUT / "participation" / f"rate{rate}" / method / f"seed{seed}"
            margs = M.CROSS_METHODS[method][1]
            cmd = M.COMMON + M.ds_args(TGT, M.CROSS_DS[TGT]) + margs + [
                "--output_dir", str(run_dir), "--seed", str(seed),
                "--init_adapter_path", str(M.OUT / "pretrain" / f"source_{SRC}_seed{seed}_adapter.pt"),
                "--client_participation", rate]
            M.run_one(cmd, run_dir, dict(stage="participation", group=f"rate{rate}", method=method,
                                         dataset=TGT, source=SRC, target=TGT, seed=seed))
print("[gaps] ALL DONE", flush=True)
