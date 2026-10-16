"""Master v7 driver: after sweeps finish, run refills, budget comparison, and the
final main matrix with the validation-selected config.

Final FedPASA framework config (validation-selected on Music->Movies):
  alpha=0.85, cold_power(kappa)=0.5, condition_scale(lambda)=<from validsweep2>, rho=0.2
Budget: rounds=12, local_steps=60 (2.5x).
"""
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, '.')
import run_www_matrix as M

ROOT = Path('.')
SEEDS = [42, 123, 456, 789, 2024]
STUDY_SEEDS = [42, 123, 456]
ALPHA, KAPPA, RHO = "0.85", "0.5", "0.3"
LAMBDA = None  # decided from validsweep2


def wait_for(cond, what):
    while not cond():
        print(f"[v7] waiting for {what}...", flush=True)
        time.sleep(60)


def count(sub):
    d = M.OUT / sub
    return len(list(d.rglob("results.json"))) if d.exists() else 0


def best_lambda():
    import json
    from statistics import mean
    best, bestv = None, -1
    for gdir in (M.OUT / "validsweep2").iterdir():
        vals = []
        for res in gdir.glob("seed*/results.json"):
            mv = json.load(open(res)).get("metrics_valid") or {}
            if mv.get("s99_HR@20") is not None:
                vals.append(mv["s99_HR@20"])
        if vals and mean(vals) > bestv:
            bestv, best = mean(vals), gdir.name
    lam = best.split("_")[0].lstrip("l")
    print(f"[v7] validation-selected lambda = {lam} (valid {bestv:.4f})", flush=True)
    return lam


def pasa_args(lam):
    return ["--method", "fedbat", "--fedbat_alpha", ALPHA, "--fedbat_cold_power", KAPPA,
            "--fedbat_condition_scale", lam, "--semantic_anchor_auto", "1",
            "--semantic_anchor_weight", RHO, "--semantic_anchor_auto_low", "1.0",
            "--semantic_anchor_auto_high", "1.5"]


BUDGETS = {"music": dict(rounds=12, steps=60), "movies": dict(rounds=12, steps=60),
           "books": dict(rounds=8, steps=30)}


def ds_args_v7(name, cfg):
    cfg = dict(cfg)
    cfg.update(BUDGETS.get(name, dict(rounds=20, steps=60)))
    return M.ds_args(name, cfg)


def run(cmd, run_dir, row=None):
    return M.run_one(cmd, run_dir, row or {})


def rerun_family(lam):
    pasa = pasa_args(lam)
    variants = {
        "fedpasa": pasa,
        "pasa_no_routing": pasa + ["--fedbat_use_cold", "0"],
        "pasa_no_condition": pasa + ["--fedbat_condition_scale", "0"],
        "pasa_no_anchor": ["--method", "fedbat", "--fedbat_alpha", ALPHA, "--fedbat_cold_power", KAPPA,
                           "--fedbat_condition_scale", lam, "--semantic_anchor_auto", "0",
                           "--semantic_anchor_weight", "0"],
        "pasa_const_routing": pasa + ["--fedbat_routing", "constant"],
        "pasa_rand_routing": pasa + ["--fedbat_routing", "random"],
        "pasa_item_anchor": pasa + ["--anchor_mode", "item"],
    }
    baselines = ["fcf", "scratch", "fedprox", "fedadam", "transfr", "l2sp", "fedper", "pfedme", "fedmoe"]
    V7 = M.OUT / "v7"
    # cross: baselines + variants, all at the SAME new budget
    for source, target in M.DIRECTIONS:
        for method in baselines + list(variants):
            if method in variants:
                margs = variants[method]
                needs_init = method != "pasa_no_anchor" or True
            else:
                needs_init, margs = M.CROSS_METHODS[method]
            for seed in SEEDS:
                run_dir = V7 / "cross" / f"{source}_to_{target}" / method / f"seed{seed}"
                cmd = M.COMMON + ds_args_v7(target, M.CROSS_DS[target]) + margs + [
                    "--output_dir", str(run_dir), "--seed", str(seed)]
                if needs_init and method not in ("fcf", "scratch", "fedprox", "fedadam"):
                    cmd += ["--init_adapter_path",
                            str(M.OUT / "pretrain" / f"source_{source}_seed{seed}_adapter.pt")]
                run(cmd, run_dir, dict(stage="v7cross", group="cross", method=method,
                                       dataset=target, source=source, target=target, seed=seed))
    # ditto (the deployed FedPASA): cross + indomain
    for source, target in M.DIRECTIONS:
        for seed in SEEDS:
            run_dir = V7 / "ditto" / f"{source}_to_{target}" / "pasa_ditto" / f"seed{seed}"
            cmd = M.COMMON + ds_args_v7(target, M.CROSS_DS[target]) + pasa + [
                "--ditto_steps", "40", "--ditto_mu", "0.15",
                "--output_dir", str(run_dir), "--seed", str(seed),
                "--init_adapter_path", str(M.OUT / "pretrain" / f"source_{source}_seed{seed}_adapter.pt")]
            run(cmd, run_dir, dict(stage="v7ditto", group="cross", method="pasa_ditto",
                                   dataset=target, source=source, target=target, seed=seed))
    # indomain: ALL methods at the same new budget
    indomain_baselines = ["fcf", "scratch", "fedprox", "fedadam", "fedper", "pfedme", "fedmoe"]
    for ds, cfg in M.INDOMAIN_DS.items():
        for method in indomain_baselines + ["fedpasa", "pasa_ditto"]:
            if method == "pasa_ditto":
                margs = pasa + ["--ditto_steps", "40", "--ditto_mu", "0.15"]
                group = "pasa_ditto"
            elif method == "fedpasa":
                margs = pasa
                group = "fedpasa"
            else:
                _, margs = M.INDOMAIN_METHODS[method]
                group = method
            for seed in SEEDS:
                run_dir = V7 / "indomain" / ds / group / f"seed{seed}"
                cmd = M.COMMON + ds_args_v7(ds, cfg) + margs + [
                    "--output_dir", str(run_dir), "--seed", str(seed)]
                run(cmd, run_dir, dict(stage="v7indomain", group="indomain", method=group,
                                       dataset=ds, source="", target=ds, seed=seed))
    # nosrc at new config (for SG consistency)
    for source, target in M.DIRECTIONS:
        for seed in SEEDS:
            run_dir = V7 / "nosrc" / f"{source}_to_{target}" / "pasa_nosrc" / f"seed{seed}"
            cmd = M.COMMON + ds_args_v7(target, M.CROSS_DS[target]) + pasa + [
                "--output_dir", str(run_dir), "--seed", str(seed)]
            run(cmd, run_dir, dict(stage="v7nosrc", group="nosrc", method="pasa_nosrc",
                                   dataset=target, source=source, target=target, seed=seed))
    print("[v7] family rerun DONE", flush=True)


def refills():
    # fixedcompute clients50 + participation (old config, robustness studies)
    for nclients, steps in ((50, 3),):
        for method in ("transfr", "fedpasa"):
            for seed in STUDY_SEEDS:
                run_dir = M.OUT / "fixedcompute" / f"clients{nclients}" / method / f"seed{seed}"
                margs = M.CROSS_METHODS[method][1]
                cfg = dict(M.CROSS_DS["movies"]); cfg["clients"] = nclients; cfg["steps"] = steps
                cmd = M.COMMON + M.ds_args("movies", cfg) + margs + [
                    "--output_dir", str(run_dir), "--seed", str(seed),
                    "--init_adapter_path", str(M.OUT / "pretrain" / f"source_music_seed{seed}_adapter.pt")]
                run(cmd, run_dir, dict(stage="fixedcompute", group=f"clients{nclients}", method=method,
                                       dataset="movies", source="music", target="movies", seed=seed))
    for rate in ("0.2", "0.5"):
        for method in ("transfr", "fedpasa"):
            for seed in STUDY_SEEDS:
                run_dir = M.OUT / "participation" / f"rate{rate}" / method / f"seed{seed}"
                margs = M.CROSS_METHODS[method][1]
                cmd = M.COMMON + M.ds_args("movies", M.CROSS_DS["movies"]) + margs + [
                    "--output_dir", str(run_dir), "--seed", str(seed),
                    "--init_adapter_path", str(M.OUT / "pretrain" / f"source_music_seed{seed}_adapter.pt"),
                    "--client_participation", rate]
                run(cmd, run_dir, dict(stage="participation", group=f"rate{rate}", method=method,
                                       dataset="movies", source="music", target="movies", seed=seed))
    print("[v7] refills DONE", flush=True)


def budget_compare():
    for method, extra in (("transfr", ["--method", "transfr"]), ("fedmoe", ["--method", "fedmoe"])):
        for seed in STUDY_SEEDS:
            run_dir = Path("outputs/budget_test") / f"mm_{method}_big" / f"seed{seed}"
            cmd = M.COMMON + M.ds_args("movies", dict(M.CROSS_DS["movies"], rounds=16, steps=150)) + extra + [
                "--output_dir", str(run_dir), "--seed", str(seed),
                "--init_adapter_path", str(M.OUT / "pretrain" / f"source_music_seed{seed}_adapter.pt")]
            run(cmd, run_dir, None)
    print("[v7] budget compare DONE", flush=True)


def studies_v7(lam):
    pasa = pasa_args(lam)
    V7 = M.OUT / "v7"
    src, tgt = "music", "movies"
    tcfg = M.CROSS_DS[tgt]
    # scalability
    for nclients in (5, 20, 50):
        for method, margs in (("transfr", M.CROSS_METHODS["transfr"][1]), ("fedpasa", pasa)):
            for seed in STUDY_SEEDS:
                run_dir = V7 / "scalability" / f"clients{nclients}" / method / f"seed{seed}"
                cfg = dict(tcfg); cfg["clients"] = nclients
                cmd = M.COMMON + ds_args_v7(tgt, cfg) + margs + [
                    "--output_dir", str(run_dir), "--seed", str(seed),
                    "--init_adapter_path", str(M.OUT / "pretrain" / f"source_{src}_seed{seed}_adapter.pt")]
                run(cmd, run_dir, dict(stage="v7scal", group=f"clients{nclients}", method=method,
                                       dataset=tgt, source=src, target=tgt, seed=seed))
    # noniid
    for split, alpha in (("dirichlet_activity", "0.1"), ("dirichlet_activity", "0.5"), ("dirichlet_popularity", "0.5")):
        for method, margs in (("transfr", M.CROSS_METHODS["transfr"][1]), ("fedpasa", pasa)):
            for seed in STUDY_SEEDS:
                run_dir = V7 / "noniid" / f"{split}_a{alpha}" / method / f"seed{seed}"
                cmd = M.COMMON + ds_args_v7(tgt, tcfg) + margs + [
                    "--output_dir", str(run_dir), "--seed", str(seed),
                    "--client_split", split, "--client_dirichlet_alpha", alpha,
                    "--init_adapter_path", str(M.OUT / "pretrain" / f"source_{src}_seed{seed}_adapter.pt")]
                run(cmd, run_dir, dict(stage="v7noniid", group=f"{split}_a{alpha}", method=method,
                                       dataset=tgt, source=src, target=tgt, seed=seed))
    # robustness
    for noise in ("0.1", "0.2", "0.3"):
        for source, target in M.DIRECTIONS:
            for method, margs in (("transfr", M.CROSS_METHODS["transfr"][1]), ("fedpasa", pasa)):
                for seed in STUDY_SEEDS:
                    run_dir = V7 / "robustness" / f"noise{noise}" / f"{source}_to_{target}" / method / f"seed{seed}"
                    cmd = M.COMMON + ds_args_v7(target, M.CROSS_DS[target]) + margs + [
                        "--output_dir", str(run_dir), "--seed", str(seed), "--update_noise", noise,
                        "--init_adapter_path", str(M.OUT / "pretrain" / f"source_{source}_seed{seed}_adapter.pt")]
                    run(cmd, run_dir, dict(stage="v7robust", group=f"noise{noise}", method=method,
                                           dataset=target, source=source, target=target, seed=seed))
    # sensitivity around the NEW operating point
    sweeps = [("alpha", "--fedbat_alpha", ["0.55", "0.7", "1.0"]),
              ("coldpower", "--fedbat_cold_power", ["0.35", "0.7", "1.0"]),
              ("condscale", "--fedbat_condition_scale", ["0.0", "0.1", "0.4"]),
              ("anchorw", "--semantic_anchor_weight", ["0.0", "0.1", "0.2", "0.5"])]
    for name, flag, values in sweeps:
        for v in values:
            for seed in STUDY_SEEDS:
                run_dir = V7 / "sensitivity" / f"{name}{v}" / f"seed{seed}"
                cmd = M.COMMON + ds_args_v7(tgt, tcfg) + list(pasa) + [flag, v,
                    "--output_dir", str(run_dir), "--seed", str(seed),
                    "--init_adapter_path", str(M.OUT / "pretrain" / f"source_{src}_seed{seed}_adapter.pt")]
                run(cmd, run_dir, dict(stage="v7sens", group=f"{name}{v}", method="fedpasa",
                                       dataset=tgt, source=src, target=tgt, seed=seed))
    # encoder (LLaMA features)
    llama = Path("features_llama")
    for source, target in [("music", "movies"), ("books", "movies"), ("music", "books")]:
        for seed in STUDY_SEEDS:
            pre_dir = V7 / "encoder" / f"pretrain_{source}_llama_seed{seed}"
            adapter_path = V7 / "encoder" / f"pretrain_{source}_llama_seed{seed}_adapter.pt"
            cfg = M.CROSS_DS[source]
            cmd = M.COMMON + ds_args_v7(source, cfg) + [
                "--precomputed_text_features", str(llama / f"{source}_llama2_7b_proj512.pt"),
                "--method", "transfr", "--output_dir", str(pre_dir), "--seed", str(seed),
                "--save_global_adapter_path", str(adapter_path)]
            run(cmd, pre_dir, None)
            for method, margs in (("transfr", M.CROSS_METHODS["transfr"][1]), ("fedpasa", pasa)):
                run_dir = V7 / "encoder" / f"{source}_to_{target}" / f"llama_{method}" / f"seed{seed}"
                cmd = M.COMMON + ds_args_v7(target, M.CROSS_DS[target]) + [
                    "--precomputed_text_features", str(llama / f"{target}_llama2_7b_proj512.pt"),
                    *margs, "--output_dir", str(run_dir), "--seed", str(seed),
                    "--init_adapter_path", str(adapter_path)]
                run(cmd, run_dir, dict(stage="v7enc", group="llama", method=method,
                                       dataset=target, source=source, target=target, seed=seed))
    # refinement variants at new config
    for vname, extra in (("pasa_ll", ["--fedbat_learn_lambda", "1"]),
                         ("pasa_cd", ["--semantic_anchor_cold_delta", "0.1"]),
                         ("pasa_both", ["--fedbat_learn_lambda", "1", "--semantic_anchor_cold_delta", "0.1"])):
        for source, target in M.DIRECTIONS:
            for seed in SEEDS:
                run_dir = V7 / "enhance" / f"{source}_to_{target}" / vname / f"seed{seed}"
                cmd = M.COMMON + ds_args_v7(target, M.CROSS_DS[target]) + pasa + extra + [
                    "--output_dir", str(run_dir), "--seed", str(seed),
                    "--init_adapter_path", str(M.OUT / "pretrain" / f"source_{source}_seed{seed}_adapter.pt")]
                run(cmd, run_dir, dict(stage="v7enh", group=vname, method=vname,
                                       dataset=target, source=source, target=target, seed=seed))
    print("[v7] studies DONE", flush=True)


def main():
    lam = best_lambda()
    refills()
    budget_compare()
    rerun_family(lam)
    studies_v7(lam)
    print("[v7] ALL DONE", flush=True)


if __name__ == "__main__":
    main()
