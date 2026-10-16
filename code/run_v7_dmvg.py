"""v7 in-domain supplement: digital_music + video_games at v7 config (all methods)."""
import sys
from pathlib import Path
sys.path.insert(0, '.')
import run_www_matrix as M

ALPHA, KAPPA, RHO, LAM = "0.85", "0.5", "0.3", "0.2"
PASA = ["--method","fedbat","--fedbat_alpha",ALPHA,"--fedbat_cold_power",KAPPA,
        "--fedbat_condition_scale",LAM,"--semantic_anchor_auto","1",
        "--semantic_anchor_weight",RHO,"--semantic_anchor_auto_low","1.0","--semantic_anchor_auto_high","1.5"]
DS = {
    "digital_music": dict(dat=Path("data/amazon_v2/digital_music/digital_music.dat"),
                          titles=Path("data/amazon_v2/digital_music/titles.json"),
                          clients=10, rounds=20, steps=60, eval_users=0),
    "video_games": dict(dat=Path("data/amazon_v2/video_games/video_games.dat"),
                        titles=Path("data/amazon_v2/video_games/titles.json"),
                        clients=10, rounds=20, steps=60, eval_users=0),
}
methods = ["fcf","scratch","fedprox","fedadam","fedper","pfedme","fedmoe","fedpasa"]
for ds, cfg in DS.items():
    for method in methods + ["pasa_ditto"]:
        if method == "pasa_ditto":
            margs = PASA + ["--ditto_steps","40","--ditto_mu","0.15"]
        elif method == "fedpasa":
            margs = PASA
        else:
            _, margs = M.INDOMAIN_METHODS[method]
        for seed in M.SEEDS:
            run_dir = M.OUT / "v7" / "indomain" / ds / method / f"seed{seed}"
            cmd = M.COMMON + M.ds_args(ds, cfg) + margs + ["--output_dir", str(run_dir), "--seed", str(seed)]
            M.run_one(cmd, run_dir, dict(stage="v7indomain", group="indomain", method=method,
                                         dataset=ds, source="", target=ds, seed=seed))
print("[v7 dm+vg] ALL DONE")
