"""FeDecider-style baseline: LLaMA-2-7B features + direction/magnitude decoupled
federated aggregation (their published mechanism: share only directional components
of low-rank updates across clients). Official repo contains README only.
6 directions x 3 seeds."""
import sys
from pathlib import Path
sys.path.insert(0, '.')
import run_www_matrix as M

LLAMA = Path("features_llama")
SEEDS = [42, 123, 456]
ARGS = ["--method", "transfr", "--aggregation", "decoupled", "--decoupled_profile_weight", "0.2",
        "--decoupled_magnitude_weight", "0.2", "--semantic_temperature", "0.1", "--semantic_self_weight", "0.2"]

for source, target in M.DIRECTIONS:
    for seed in SEEDS:
        # llama source pretrain (shared)
        pre_dir = M.OUT / "fedecider" / f"pretrain_{source}_llama_seed{seed}"
        adapter_path = M.OUT / "fedecider" / f"pretrain_{source}_llama_seed{seed}_adapter.pt"
        cfg = M.CROSS_DS[source]
        cmd = M.COMMON + M.ds_args(source, cfg) + [
            "--precomputed_text_features", str(LLAMA / f"{source}_llama2_7b_proj512.pt"),
            "--method", "transfr", "--output_dir", str(pre_dir), "--seed", str(seed),
            "--save_global_adapter_path", str(adapter_path)]
        M.run_one(cmd, pre_dir, None)
        run_dir = M.OUT / "fedecider" / f"{source}_to_{target}" / "fedecider" / f"seed{seed}"
        cmd = M.COMMON + M.ds_args(target, M.CROSS_DS[target]) + [
            "--precomputed_text_features", str(LLAMA / f"{target}_llama2_7b_proj512.pt"),
            *ARGS, "--output_dir", str(run_dir), "--seed", str(seed),
            "--init_adapter_path", str(adapter_path)]
        M.run_one(cmd, run_dir, dict(stage="fedecider", group="cross", method="fedecider",
                                     dataset=target, source=source, target=target, seed=seed))
print("[fedecider] ALL DONE")
