import sys
from pathlib import Path
sys.path.insert(0, '.')
import run_www_matrix as M
VG = dict(dat=Path("data/amazon_v2/video_games/video_games.dat"),
          titles=Path("data/amazon_v2/video_games/titles.json"),
          clients=10, rounds=20, steps=40, eval_users=0)
for method, (needs_init, margs) in M.INDOMAIN_METHODS.items():
    for seed in M.SEEDS:
        run_dir = M.OUT / "indomain" / "video_games" / method / f"seed{seed}"
        cmd = M.COMMON + M.ds_args("video_games", VG) + margs + [
            "--output_dir", str(run_dir), "--seed", str(seed)]
        M.run_one(cmd, run_dir, dict(stage="indomain", group="indomain", method=method,
                                     dataset="video_games", source="", target="video_games", seed=seed))
print("[vg] ALL DONE")
