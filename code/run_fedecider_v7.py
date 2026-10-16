"""FeDecider-style at the v7 budget (rounds 12x60 music/movies, 8x30 books)."""
import argparse
import sys
from pathlib import Path

import torch

sys.path.insert(0, '.')
import run_www_matrix as M

parser = argparse.ArgumentParser()
parser.add_argument("--output-root", type=Path, default=None,
                    help="Independent output root; defaults to the paper's v7 output tree.")
parser.add_argument("--seeds", type=int, nargs="+", default=[42, 123, 456])
args = parser.parse_args()

if not torch.cuda.is_available():
    raise RuntimeError("CUDA is required for this reproduction; CPU fallback is disabled.")
if args.output_root is not None:
    output_root = args.output_root
    if not output_root.is_absolute():
        output_root = M.ROOT / output_root
    M.OUT = output_root.resolve()
    M.LEDGER = M.OUT / "ledger.csv"

BUDGETS = {"music": dict(rounds=12, steps=60), "movies": dict(rounds=12, steps=60), "books": dict(rounds=8, steps=30)}
LLAMA = Path("features_llama")
SEEDS = args.seeds
ARGS = ["--method", "transfr", "--aggregation", "decoupled", "--decoupled_profile_weight", "0.2",
        "--decoupled_magnitude_weight", "0.2", "--semantic_temperature", "0.1", "--semantic_self_weight", "0.2"]
def ds_args_v7(name, cfg):
    cfg = dict(cfg); cfg.update(BUDGETS.get(name, dict(rounds=20, steps=60)))
    return M.ds_args(name, cfg)
for source, target in M.DIRECTIONS:
    for seed in SEEDS:
        pre_dir = M.OUT / "v7" / "fedecider" / f"pretrain_{source}_llama_seed{seed}"
        adapter_path = M.OUT / "v7" / "fedecider" / f"pretrain_{source}_llama_seed{seed}_adapter.pt"
        cfg = M.CROSS_DS[source]
        cmd = M.COMMON + ds_args_v7(source, cfg) + [
            "--precomputed_text_features", str(LLAMA / f"{source}_llama2_7b_proj512.pt"),
            "--method", "transfr", "--output_dir", str(pre_dir), "--seed", str(seed),
            "--save_global_adapter_path", str(adapter_path)]
        M.run_one(cmd, pre_dir, None)
        run_dir = M.OUT / "v7" / "fedecider" / f"{source}_to_{target}" / "fedecider" / f"seed{seed}"
        cmd = M.COMMON + ds_args_v7(target, M.CROSS_DS[target]) + [
            "--precomputed_text_features", str(LLAMA / f"{target}_llama2_7b_proj512.pt"),
            *ARGS, "--output_dir", str(run_dir), "--seed", str(seed),
            "--init_adapter_path", str(adapter_path)]
        M.run_one(cmd, run_dir, dict(stage="v7fedecider", group="cross", method="fedecider",
                                     dataset=target, source=source, target=target, seed=seed))
print("[fedecider-v7] ALL DONE")
