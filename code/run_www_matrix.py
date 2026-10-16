"""WWW'27 rebuild orchestrator for FedPASA.

One uniform, fully disclosed protocol. Every number in the new paper comes
from this matrix; every run writes results.json with full args and metrics
(full-ranking primary, sampled-99 secondary, plus bucketed slices and drift).

Stages:
    python run_www_matrix.py --stage features     # encode text features once
    python run_www_matrix.py --stage pretrain     # source-domain pretraining
    python run_www_matrix.py --stage cross        # cross-domain main matrix
    python run_www_matrix.py --stage indomain     # in-domain main matrix
    python run_www_matrix.py --stage studies      # scalability/noniid/robustness/sensitivity
    python run_www_matrix.py --stage all
"""
from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
RUNNER = ROOT / "run_transfr_style.py"
OUT = ROOT / "outputs" / "www_v1"
FEAT = ROOT / "features"
LEDGER = OUT / "ledger.csv"

SEEDS = [42, 123, 456, 789, 2024]
STUDY_SEEDS = [42, 123, 456]

CDR = ROOT / "data" / "amazon_v2_transfr20_30"
CROSS_DS = {
    "music": dict(dat=CDR / "cds_and_vinyl/cds_and_vinyl.dat",
                  titles=CDR / "cds_and_vinyl/titles_real_metadata.json",
                  clients=5, rounds=8, steps=30, eval_users=5000),
    "movies": dict(dat=CDR / "movies_and_tv/movies_and_tv.dat",
                   titles=CDR / "movies_and_tv/titles_real_metadata.json",
                   clients=5, rounds=8, steps=30, eval_users=5000),
    "books": dict(dat=CDR / "books/books.dat",
                  titles=CDR / "books/titles_real_metadata.json",
                  clients=5, rounds=6, steps=20, eval_users=5000),
}
INDOMAIN_DS = {
    "ml100k": dict(dat=ROOT / "data/indomain_local/fedcia_ml100k_100/ml100k_100.dat",
                   titles=ROOT / "data/indomain_titles/ml100k_titles.json",
                   clients=10, rounds=20, steps=40, eval_users=0),
    "ku": dict(dat=ROOT / "data/indomain_local/ku/ku.dat",
               titles=None, features=ROOT / "data/indomain_local/ku/text_features.npy",
               clients=5, rounds=20, steps=40, eval_users=0),
    "luxury_beauty": dict(dat=ROOT / "data/amazon_v2/luxury_beauty/luxury_beauty.dat",
                          titles=ROOT / "data/amazon_v2/luxury_beauty/titles.json",
                          clients=10, rounds=20, steps=40, eval_users=0),
    "ml1m": dict(dat=ROOT / "data/indomain_local/fedcia_ml1m_100/ml1m_100.dat",
                 titles=ROOT / "data/indomain_titles/ml1m_titles_aligned.json",
                 clients=10, rounds=20, steps=40, eval_users=0),
}

COMMON = [
    "--text_model", "distilbert-base-uncased",
    "--title_source", "fallback",
    "--adapter_dim", "64", "--adapter_layers", "2", "--dropout", "0.1",
    "--lr", "1e-3", "--batch_size", "32", "--negatives", "3",
    "--loss", "bpr", "--logit_scale", "10", "--weight_decay", "1e-4",
    "--cold_item_threshold", "5",
    "--head_steps", "0",
    "--save_artifacts", "0",
    "--skip_valid_eval",
    "--eval_sampled_negatives", "99",
    "--metric_ks", "5", "10", "20",
]

FEDPASA = [
    "--method", "fedbat",
    "--fedbat_alpha", "0.55", "--fedbat_cold_power", "0.7",
    "--fedbat_condition_scale", "0.4",
    "--semantic_anchor_auto", "1", "--semantic_anchor_weight", "0.2",
    "--semantic_anchor_auto_low", "1.0", "--semantic_anchor_auto_high", "1.5",
]

# name -> (needs_source_init, extra args)
CROSS_METHODS = {
    "fcf":               (False, ["--only_mf_baseline", "--mf_rounds", "25", "--mf_local_steps", "80"]),
    "scratch":           (False, ["--method", "transfr"]),
    "fedprox":           (False, ["--method", "transfr", "--prox_mu", "0.1"]),
    "fedadam":           (False, ["--method", "transfr", "--server_optimizer", "fedadam", "--server_lr", "0.01"]),
    "transfr":           (True,  ["--method", "transfr"]),
    "l2sp":              (True,  ["--method", "transfr", "--l2sp_mu", "0.01"]),
    "fedper":            (True,  ["--method", "dual", "--dual_personal_mix", "0.5"]),
    "pfedme":            (True,  ["--method", "transfr", "--ditto_steps", "40", "--ditto_mu", "0.15"]),
    "fedmoe":            (True,  ["--method", "fedmoe"]),
    "fedpasa":           (True,  FEDPASA),
    "pasa_no_routing":   (True,  FEDPASA + ["--fedbat_use_cold", "0"]),
    "pasa_no_condition": (True,  FEDPASA + ["--fedbat_condition_scale", "0"]),
    "pasa_no_anchor":    (True,  ["--method", "fedbat", "--fedbat_alpha", "0.55", "--fedbat_cold_power", "0.7",
                                  "--fedbat_condition_scale", "0.4",
                                  "--semantic_anchor_auto", "0", "--semantic_anchor_weight", "0"]),
    "pasa_const_routing": (True, FEDPASA + ["--fedbat_routing", "constant"]),
    "pasa_rand_routing":  (True, FEDPASA + ["--fedbat_routing", "random"]),
    "pasa_item_anchor":   (True, FEDPASA + ["--anchor_mode", "item"]),
}

INDOMAIN_METHODS = {k: (False, v[1]) for k, v in CROSS_METHODS.items()
                    if k not in ("transfr", "l2sp")}

DIRECTIONS = [("music", "books"), ("music", "movies"), ("books", "movies"),
              ("books", "music"), ("movies", "books"), ("movies", "music")]

LEDGER_FIELDS = ["stage", "group", "method", "dataset", "source", "target", "seed",
                 "HR@5", "NDCG@5", "HR@10", "NDCG@10", "HR@20", "NDCG@20",
                 "s99_HR@20", "s99_NDCG@20", "drift_cos", "drift_cos_cold", "drift_cos_warm",
                 "freq_zero_HR@20", "freq_1_5_HR@20", "freq_6_20_HR@20", "freq_20plus_HR@20",
                 "freq_zero_NDCG@20", "freq_1_5_NDCG@20", "freq_6_20_NDCG@20", "freq_20plus_NDCG@20",
                 "cold_HR@20", "warm_HR@20", "runtime_s", "run_dir"]


def metrics_of(run_dir: Path, method: str) -> dict:
    p = run_dir / "results.json"
    if not p.exists():
        return {}
    d = json.loads(p.read_text(encoding="utf-8"))
    if method == "fcf":
        m = d.get("metrics_fedmf") or {}
    elif method == "pfedme":
        m = d.get("metrics_ditto") or d.get("metrics_no_pap") or {}
    else:
        m = d.get("metrics_no_pap") or {}
    m = {k: v for k, v in m.items() if isinstance(v, (int, float))}
    m["runtime_s"] = round(d.get("runtime", {}).get("total_seconds", 0.0), 1)
    return m


def append_ledger(row: dict):
    LEDGER.parent.mkdir(parents=True, exist_ok=True)
    new = not LEDGER.exists()
    with LEDGER.open("a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=LEDGER_FIELDS, extrasaction="ignore")
        if new:
            w.writeheader()
        w.writerow(row)


def run_one(cmd: list[str], run_dir: Path, ledger_row: dict, max_attempts: int = 3) -> bool:
    if (run_dir / "results.json").exists():
        print("SKIP", run_dir.name)
        if ledger_row is not None:
            m = metrics_of(run_dir, ledger_row.get("method", ""))
            if m:
                append_ledger({**ledger_row, **m, "run_dir": str(run_dir)})
        return True
    import os
    env = dict(os.environ)
    env["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    for attempt in range(1, max_attempts + 1):
        run_dir.mkdir(parents=True, exist_ok=True)
        (run_dir / "command.txt").write_text(" ".join(map(str, cmd)) + "\n", encoding="utf-8")
        t0 = time.time()
        with (run_dir / "run.log").open("w", encoding="utf-8") as log:
            rc = subprocess.call([sys.executable, "-u", str(RUNNER)] + cmd, cwd=ROOT,
                                 stdout=log, stderr=subprocess.STDOUT, env=env)
        ok = rc == 0 and (run_dir / "results.json").exists()
        print(("OK  " if ok else "FAIL"), run_dir.name, f"{time.time()-t0:.0f}s",
              "" if ok else f"(attempt {attempt}/{max_attempts})")
        if ok:
            break
        time.sleep(10)
    if ok and ledger_row is not None:
        append_ledger({**ledger_row, **metrics_of(run_dir, ledger_row.get("method", "")),
                       "run_dir": str(run_dir)})
    return ok


def ds_args(name, cfg):
    args = ["--data_path", str(cfg["dat"]), "--dataset_name", name,
            "--num_clients", str(cfg["clients"]), "--rounds", str(cfg["rounds"]),
            "--local_steps", str(cfg["steps"]), "--eval_users", str(cfg["eval_users"])]
    if cfg.get("features"):
        args += ["--precomputed_text_features", str(cfg["features"])]
    else:
        args += ["--item_titles_json", str(cfg["titles"])]
        feat = FEAT / f"{name}.pt"
        if feat.exists():
            args += ["--precomputed_text_features", str(feat)]
    return args


def stage_features():
    FEAT.mkdir(exist_ok=True)
    for name, cfg in {**CROSS_DS, **INDOMAIN_DS}.items():
        feat = FEAT / f"{name}.pt"
        if feat.exists() or cfg.get("features"):
            continue
        code = (
            "import sys; sys.path.insert(0, r'%s');"
            "from run_transfr_style import build_text_features;"
            "import json, torch;"
            "from pathlib import Path;"
            "raw = json.loads(Path(r'%s').read_text(encoding='utf-8'));"
            "titles = {int(k): str(v) for k, v in raw.items()};"
            "feats = build_text_features(titles, 'distilbert-base-uncased', Path(r'%s'), 64,"
            " torch.device('cuda' if torch.cuda.is_available() else 'cpu'));"
            "print(feats.shape)"
        ) % (ROOT, cfg["titles"], feat)
        print("ENCODE", name)
        subprocess.call([sys.executable, "-c", code], cwd=ROOT)


def stage_pretrain(seeds):
    for source in ("music", "books", "movies"):
        for seed in seeds:
            run_dir = OUT / "pretrain" / f"source_{source}_seed{seed}"
            adapter_path = OUT / "pretrain" / f"source_{source}_seed{seed}_adapter.pt"
            cmd = COMMON + ds_args(source, CROSS_DS[source]) + [
                "--method", "transfr", "--output_dir", str(run_dir), "--seed", str(seed),
                "--save_global_adapter_path", str(adapter_path)]
            run_one(cmd, run_dir, dict(stage="pretrain", group="pretrain", method="scratch",
                                       dataset=source, source=source, target="", seed=seed))


def stage_cross(seeds, methods=None, directions=None):
    methods = methods or list(CROSS_METHODS)
    directions = directions or DIRECTIONS
    for source, target in directions:
        for method in methods:
            needs_init, margs = CROSS_METHODS[method]
            for seed in seeds:
                run_dir = OUT / "cross" / f"{source}_to_{target}" / method / f"seed{seed}"
                cmd = COMMON + ds_args(target, CROSS_DS[target]) + margs + [
                    "--output_dir", str(run_dir), "--seed", str(seed)]
                if needs_init:
                    cmd += ["--init_adapter_path",
                            str(OUT / "pretrain" / f"source_{source}_seed{seed}_adapter.pt")]
                run_one(cmd, run_dir, dict(stage="cross", group="cross", method=method,
                                           dataset=target, source=source, target=target, seed=seed))


def stage_indomain(seeds, methods=None, datasets=None):
    methods = methods or list(INDOMAIN_METHODS)
    for ds, cfg in INDOMAIN_DS.items():
        if datasets and ds not in datasets:
            continue
        for method in methods:
            needs_init, margs = INDOMAIN_METHODS[method]
            for seed in seeds:
                run_dir = OUT / "indomain" / ds / method / f"seed{seed}"
                cmd = COMMON + ds_args(ds, cfg) + margs + [
                    "--output_dir", str(run_dir), "--seed", str(seed)]
                run_one(cmd, run_dir, dict(stage="indomain", group="indomain", method=method,
                                           dataset=ds, source="", target=ds, seed=seed))


def stage_studies(seeds):
    src, tgt = "music", "movies"
    # scalability: number of clients
    for nclients in (5, 20, 50):
        for method in ("transfr", "fedpasa"):
            needs_init, margs = CROSS_METHODS[method]
            for seed in seeds:
                run_dir = OUT / "scalability" / f"clients{nclients}" / method / f"seed{seed}"
                cfg = dict(CROSS_DS[tgt]); cfg["clients"] = nclients
                cmd = COMMON + ds_args(tgt, cfg) + margs + [
                    "--output_dir", str(run_dir), "--seed", str(seed),
                    "--init_adapter_path", str(OUT / "pretrain" / f"source_{src}_seed{seed}_adapter.pt")]
                run_one(cmd, run_dir, dict(stage="scalability", group=f"clients{nclients}",
                                           method=method, dataset=tgt, source=src, target=tgt, seed=seed))
    # non-IID client partitions
    for split, alpha in (("dirichlet_activity", "0.1"), ("dirichlet_activity", "0.5"),
                         ("dirichlet_popularity", "0.5")):
        for method in ("transfr", "fedpasa"):
            needs_init, margs = CROSS_METHODS[method]
            for seed in seeds:
                run_dir = OUT / "noniid" / f"{split}_a{alpha}" / method / f"seed{seed}"
                cmd = COMMON + ds_args(tgt, CROSS_DS[tgt]) + margs + [
                    "--output_dir", str(run_dir), "--seed", str(seed),
                    "--client_split", split, "--client_dirichlet_alpha", alpha,
                    "--init_adapter_path", str(OUT / "pretrain" / f"source_{src}_seed{seed}_adapter.pt")]
                run_one(cmd, run_dir, dict(stage="noniid", group=f"{split}_a{alpha}",
                                           method=method, dataset=tgt, source=src, target=tgt, seed=seed))
    # robustness to update perturbation
    for noise in ("0.1", "0.2", "0.3"):
        for source, target in DIRECTIONS:
            for method in ("transfr", "fedpasa"):
                needs_init, margs = CROSS_METHODS[method]
                for seed in seeds:
                    run_dir = OUT / "robustness" / f"noise{noise}" / f"{source}_to_{target}" / method / f"seed{seed}"
                    cmd = COMMON + ds_args(target, CROSS_DS[target]) + margs + [
                        "--output_dir", str(run_dir), "--seed", str(seed),
                        "--update_noise", noise,
                        "--init_adapter_path", str(OUT / "pretrain" / f"source_{source}_seed{seed}_adapter.pt")]
                    run_one(cmd, run_dir, dict(stage="robustness", group=f"noise{noise}",
                                               method=method, dataset=target, source=source, target=target, seed=seed))
    # sensitivity: one factor at a time around the FedPASA defaults
    sweeps = [
        ("alpha", "--fedbat_alpha", ["0.25", "0.4", "0.7", "0.85"]),
        ("coldpower", "--fedbat_cold_power", ["0.5", "1.0", "1.5"]),
        ("condscale", "--fedbat_condition_scale", ["0.0", "0.2", "0.6"]),
        ("anchorw", "--semantic_anchor_weight", ["0.0", "0.1", "0.3", "0.5"]),
    ]
    for name, flag, values in sweeps:
        for v in values:
            for seed in seeds:
                run_dir = OUT / "sensitivity" / f"{name}{v}" / f"seed{seed}"
                cmd = COMMON + ds_args(tgt, CROSS_DS[tgt]) + list(FEDPASA) + [
                    flag, v, "--output_dir", str(run_dir), "--seed", str(seed),
                    "--init_adapter_path", str(OUT / "pretrain" / f"source_{src}_seed{seed}_adapter.pt")]
                run_one(cmd, run_dir, dict(stage="sensitivity", group=f"{name}{v}",
                                           method="fedpasa", dataset=tgt, source=src, target=tgt, seed=seed))


def stage_encoder(seeds):
    """DistilBERT vs LLaMA-2-7B(proj512) on three directions (real features for both)."""
    llama = ROOT / "features_llama"
    for source, target in [("music", "movies"), ("books", "movies"), ("music", "books")]:
        for seed in seeds:
            # source pretrain with LLaMA features
            pre_dir = OUT / "encoder" / f"pretrain_{source}_llama_seed{seed}"
            adapter_path = OUT / "encoder" / f"pretrain_{source}_llama_seed{seed}_adapter.pt"
            cfg = CROSS_DS[source]
            cmd = COMMON + ["--data_path", str(cfg["dat"]), "--dataset_name", source,
                            "--num_clients", str(cfg["clients"]), "--rounds", str(cfg["rounds"]),
                            "--local_steps", str(cfg["steps"]), "--eval_users", str(cfg["eval_users"]),
                            "--precomputed_text_features", str(llama / f"{source}_llama2_7b_proj512.pt"),
                            "--method", "transfr", "--output_dir", str(pre_dir), "--seed", str(seed),
                            "--save_global_adapter_path", str(adapter_path)]
            run_one(cmd, pre_dir, None)
            for method in ("transfr", "fedpasa"):
                margs = CROSS_METHODS[method][1]
                run_dir = OUT / "encoder" / f"{source}_to_{target}" / f"llama_{method}" / f"seed{seed}"
                cfg = CROSS_DS[target]
                cmd = COMMON + ["--data_path", str(cfg["dat"]), "--dataset_name", target,
                                "--num_clients", str(cfg["clients"]), "--rounds", str(cfg["rounds"]),
                                "--local_steps", str(cfg["steps"]), "--eval_users", str(cfg["eval_users"]),
                                "--precomputed_text_features", str(llama / f"{target}_llama2_7b_proj512.pt"),
                                *margs, "--output_dir", str(run_dir), "--seed", str(seed),
                                "--init_adapter_path", str(adapter_path)]
                run_one(cmd, run_dir, dict(stage="encoder", group="llama", method=method,
                                           dataset=target, source=source, target=target, seed=seed))


def stage_figruns():
    """Runs with saved artifacts for visualization (Music->Movies, seed 42)."""
    src, tgt, seed = "music", "movies", 42
    adapter_path = OUT / "pretrain" / f"source_{src}_seed{seed}_adapter.pt"
    for method, margs in (("transfr", CROSS_METHODS["transfr"][1]),
                          ("fedpasa", CROSS_METHODS["fedpasa"][1]),
                          ("pasa_no_anchor", CROSS_METHODS["pasa_no_anchor"][1])):
        run_dir = OUT / "figruns" / f"{src}_to_{tgt}" / method / f"seed{seed}"
        cfg = CROSS_DS[tgt]
        cmd = list(COMMON)
        cmd[cmd.index("--save_artifacts") + 1] = "1"
        cmd += ds_args(tgt, cfg) + margs + [
            "--output_dir", str(run_dir), "--seed", str(seed),
            "--init_adapter_path", str(adapter_path)]
        run_one(cmd, run_dir, None)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", required=True,
                    choices=["features", "pretrain", "cross", "indomain", "studies", "encoder", "figruns", "all"])
    ap.add_argument("--seeds", type=int, nargs="+", default=SEEDS)
    ap.add_argument("--methods", nargs="+", default=None)
    ap.add_argument("--datasets", nargs="+", default=None)
    ap.add_argument("--directions", nargs="+", default=None,
                    help="e.g. music_to_books music_to_movies")
    args = ap.parse_args()
    dirs = None
    if args.directions:
        dirs = [tuple(d.split("_to_")) for d in args.directions]
    if args.stage in ("features", "all"):
        stage_features()
    if args.stage in ("pretrain", "all"):
        stage_pretrain(args.seeds)
    if args.stage in ("cross", "all"):
        stage_cross(args.seeds, args.methods, dirs)
    if args.stage in ("indomain", "all"):
        stage_indomain(args.seeds, args.methods, args.datasets)
    if args.stage in ("studies", "all"):
        stage_studies(STUDY_SEEDS)
    if args.stage in ("encoder", "all"):
        stage_encoder(STUDY_SEEDS)
    if args.stage in ("figruns", "all"):
        stage_figruns()


if __name__ == "__main__":
    main()
