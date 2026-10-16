"""Cross-domain transfer experiment orchestrator for FedBASA (AAAI 2027).

三种方法对比:
  scratch   - 目标域从零训练 (no transfer)           [下界]
  transfr   - 源域预训练 -> FedAvg 适应目标域          [TransFR baseline]
  fedbasa   - 源域预训练 -> FedBAT+AutoAnchor 适应     [我们的方法]

CPU 快速跑 (默认, 约30min):
  python transfr_repro/run_cross_domain_transfer.py --cpu

GPU 完整跑:
  python transfr_repro/run_cross_domain_transfer.py --force_gpu --sources all --targets all --seeds 1 2 3
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
RUN_SCRIPT = ROOT / "run_transfr_style.py"
DEFAULT_OUTPUT_ROOT = ROOT / "outputs" / "transfer"
TRANSFR_DATA_ROOT = ROOT / "data" / "amazon_v2_transfr20_30"


def preferred_titles_path(dataset_dir: Path) -> Path:
    real = dataset_dir / "titles_real_metadata.json"
    if real.exists():
        return real
    return dataset_dir / "titles.json"

# 每个 Amazon 域的数据路径和 CPU-friendly 超参
DATASETS: dict[str, dict] = {
    "all_beauty": {
        "data": ROOT / "data/amazon_v2/all_beauty/all_beauty.dat",
        "titles": ROOT / "data/amazon_v2/all_beauty/titles.json",
        "num_clients": 10, "rounds": 15, "local_steps": 40, "head_steps": 40, "eval_users": 0,
    },
    "luxury_beauty": {
        "data": ROOT / "data/amazon_v2/luxury_beauty/luxury_beauty.dat",
        "titles": ROOT / "data/amazon_v2/luxury_beauty/titles.json",
        "num_clients": 15, "rounds": 20, "local_steps": 50, "head_steps": 60, "eval_users": 500,
    },
    "digital_music": {
        "data": ROOT / "data/amazon_v2/digital_music/digital_music.dat",
        "titles": ROOT / "data/amazon_v2/digital_music/titles.json",
        "num_clients": 20, "rounds": 25, "local_steps": 80, "head_steps": 80, "eval_users": 500,
    },
    "music": {
        "data": TRANSFR_DATA_ROOT / "cds_and_vinyl/cds_and_vinyl.dat",
        "titles": preferred_titles_path(TRANSFR_DATA_ROOT / "cds_and_vinyl"),
        "num_clients": 5, "rounds": 8, "local_steps": 30, "head_steps": 30, "eval_users": 500,
    },
    "books": {
        "data": TRANSFR_DATA_ROOT / "books/books.dat",
        "titles": preferred_titles_path(TRANSFR_DATA_ROOT / "books"),
        "num_clients": 3, "rounds": 6, "local_steps": 20, "head_steps": 20, "eval_users": 500,
    },
    "movies": {
        "data": TRANSFR_DATA_ROOT / "movies_and_tv/movies_and_tv.dat",
        "titles": preferred_titles_path(TRANSFR_DATA_ROOT / "movies_and_tv"),
        "num_clients": 5, "rounds": 8, "local_steps": 30, "head_steps": 30, "eval_users": 500,
    },
    "video_games": {
        "data": ROOT / "data/amazon_v2/video_games/video_games.dat",
        "titles": ROOT / "data/amazon_v2/video_games/titles.json",
        "num_clients": 20, "rounds": 25, "local_steps": 80, "head_steps": 80, "eval_users": 500,
    },
}

# 默认 CPU 下跑较小的三个域作源域
CPU_SOURCES = ["all_beauty", "luxury_beauty", "digital_music"]
ALL_DOMAINS = list(DATASETS)
TRANSFR_ALIGNED_SOURCES = ["music", "books"]
TRANSFR_ALIGNED_TARGETS = ["books", "movies"]

# 所有运行共享的基础参数
COMMON_ARGS = [
    "--text_model", "distilbert-base-uncased",
    "--title_source", "fallback",
    "--adapter_dim", "64",
    "--adapter_layers", "2",
    "--dropout", "0.1",
    "--lr", "1e-3",
    "--head_lr", "1e-3",
    "--batch_size", "32",
    "--negatives", "3",
    "--loss", "bpr",
    "--cold_item_threshold", "5",  # 用于 cold/warm 分组评测
    "--logit_scale", "10",
    "--weight_decay", "1e-4",
    "--save_artifacts", "0",
    "--eval_sampled_negatives", "99",  # sampled eval: 1 pos + 99 neg (UniSRec/SASRec protocol)
]

# 各方法特有参数（叠加在 COMMON_ARGS + 数据集参数之上）
METHOD_ARGS: dict[str, list[str]] = {
    "no_transfer": [
        "--method", "transfr",
        "--server_optimizer", "fedavg",
        "--aggregation", "fedavg",
        "--semantic_anchor_weight", "0",
        "--prox_mu", "0",
    ],
    "fcf": [
        "--method", "transfr",
        "--server_optimizer", "fedavg",
        "--aggregation", "fedavg",
        "--semantic_anchor_weight", "0",
        "--prox_mu", "0",
        "--only_mf_baseline",
        "--mf_rounds", "25",
        "--mf_local_steps", "80",
        "--mf_lr", "1e-3",
    ],
    # 从零训练，无迁移，无 FedBAT/AutoAnchor
    "scratch": [
        "--method", "transfr",
        "--server_optimizer", "fedavg",
        "--aggregation", "fedavg",
        "--semantic_anchor_weight", "0",
        "--prox_mu", "0",
    ],
    # FedProx-text 基线: 文本 adapter + 近端正则, 从零训练 (无迁移)
    "fedprox": [
        "--method", "transfr",
        "--server_optimizer", "fedavg",
        "--aggregation", "fedavg",
        "--semantic_anchor_weight", "0",
        "--prox_mu", "0.1",
    ],
    # FedAdam-text 基线: 文本 adapter + 自适应服务器优化器, 从零训练 (无迁移)
    "fedadam": [
        "--method", "transfr",
        "--server_optimizer", "fedadam",
        "--aggregation", "fedavg",
        "--semantic_anchor_weight", "0",
        "--prox_mu", "0",
    ],
    # TransFR 迁移基线: 加载预训练 adapter + 普通 FedAvg 适应
    "transfr": [
        "--method", "transfr",
        "--server_optimizer", "fedavg",
        "--aggregation", "fedavg",
        "--semantic_anchor_weight", "0",
        "--prox_mu", "0",
    ],
    # 我们的方法: 加载预训练 adapter + FedBAT(行为感知) + AutoAnchor(语义锚定)
    "fedbasa": [
        "--method", "fedbat",
        "--server_optimizer", "fedavg",
        "--aggregation", "fedavg",
        "--fedbat_alpha", "0.35",
        "--fedbat_condition_scale", "0.25",
        "--semantic_anchor_auto", "1",
        "--semantic_anchor_weight", "0.3",
        "--semantic_anchor_auto_low", "1.0",
        "--semantic_anchor_auto_high", "1.5",
        "--prox_mu", "0.1",
    ],
}

DEDICATED_ORIGINAL_METHODS = {
    "cmf",
    "emcdr",
    "disencdr",
    "ifedrec",
    "fellrec",
    "fedct",
    "feddcsr",
}
SCRATCH_METHODS = {"no_transfer", "scratch", "fcf", "fedprox", "fedadam"}
TRANSFER_METHODS = {
    "transfr",
    "fedbasa",
}


def dataset_args(name: str) -> list[str]:
    cfg = DATASETS[name]
    return [
        "--data_path", str(cfg["data"]),
        "--item_titles_json", str(cfg["titles"]),
        "--dataset_name", name,
        "--num_clients", str(cfg["num_clients"]),
        "--rounds", str(cfg["rounds"]),
        "--local_steps", str(cfg["local_steps"]),
        "--head_steps", str(cfg["head_steps"]),
        "--eval_users", str(cfg["eval_users"]),
    ]


def result_exists(run_dir: Path) -> bool:
    return (run_dir / "results.json").is_file()


def shared_feature_cache(domain: str, output_root: Path) -> Path | None:
    # Strict paper runs must encode the repaired real text source. Do not reuse
    # old transfer caches whose titles may be placeholders.
    return None
    candidates = [
        output_root / f"target_{domain}_scratch_seed1" / "item_text_features_titles_distilbert-base-uncased.pt",
        output_root / f"pretrain_{domain}_seed1" / "item_text_features_titles_distilbert-base-uncased.pt",
    ]
    for path in candidates:
        if path.exists():
            return path
    return None


def run_cmd(cmd: list, dry_run: bool) -> bool:
    cmd = [str(c) for c in cmd]
    print("\nRUN:", " ".join(cmd))
    if dry_run:
        return True
    try:
        subprocess.run(cmd, check=True)
        return True
    except subprocess.CalledProcessError as e:
        print(f"ERROR exit={e.returncode}")
        return False


def pretrain_source(source: str, seed: int, output_root: Path, use_cpu: bool, dry_run: bool) -> Path | None:
    """在源域做联邦预训练，保存 global adapter 供后续迁移使用。"""
    run_name = f"pretrain_{source}_seed{seed}"
    run_dir = output_root / run_name
    adapter_path = output_root / f"pretrain_{source}_seed{seed}_adapter.pt"

    if result_exists(run_dir) and adapter_path.exists():
        print(f"SKIP pretrain {run_name} (exists)")
        return adapter_path

    cmd = (
        [sys.executable, RUN_SCRIPT]
        + COMMON_ARGS
        + dataset_args(source)
        + METHOD_ARGS["scratch"]
        + ["--output_dir", run_dir,
           "--seed", seed,
           "--save_global_adapter_path", adapter_path]
        + (["--cpu"] if use_cpu else [])
    )
    return adapter_path if run_cmd(cmd, dry_run) else None


def adapt_target(
    source: str | None,
    target: str,
    method: str,
    seed: int,
    output_root: Path,
    use_cpu: bool,
    dry_run: bool,
    adapter_path: Path | None = None,
) -> str | None:
    """在目标域做适应（scratch / transfr / fedbasa）。"""
    prefix = f"target_{target}_from_{source}" if source else f"target_{target}"
    run_name = f"{prefix}_{method}_seed{seed}"
    run_dir = output_root / run_name

    if result_exists(run_dir):
        print(f"SKIP {run_name} (exists)")
        return run_name

    feature_cache = shared_feature_cache(target, output_root)
    cmd = (
        [sys.executable, RUN_SCRIPT]
        + COMMON_ARGS
        + dataset_args(target)
        + METHOD_ARGS[method]
        + ["--output_dir", run_dir,
           "--seed", seed]
        + (["--init_adapter_path", adapter_path] if adapter_path and adapter_path.exists() else [])
        + (["--precomputed_text_features", feature_cache] if feature_cache else [])
        + (["--cpu"] if use_cpu else [])
    )
    return run_name if run_cmd(cmd, dry_run) else None


def load_metrics(run_dir: Path, method: str | None = None) -> dict:
    p = run_dir / "results.json"
    if not p.exists():
        return {}
    data = json.loads(p.read_text(encoding="utf-8"))
    # 优先用 PAP 后的指标，fallback 到 no_pap
    if method == "fcf":
        m = data.get("metrics_fedmf") or {}
    elif method == "ditto":
        m = data.get("metrics_ditto") or data.get("metrics_pap") or data.get("metrics_no_pap") or {}
    else:
        m = data.get("metrics_pap") or data.get("metrics_no_pap") or {}
    out = {k: round(float(v), 4) for k, v in m.items() if isinstance(v, (int, float))}
    return out


def normalize_method_name(name: str) -> str:
    aliases = {
        "NoTransfer": "no_transfer",
        "No Transfer": "no_transfer",
        "FCF": "fcf",
        "TransFR": "transfr",
        "FedBASA": "fedbasa",
        "CMF": "cmf",
        "EMCDR": "emcdr",
        "DisenCDR": "disencdr",
        "IFedRec": "ifedrec",
        "FELLRec": "fellrec",
        "FedCT": "fedct",
        "FedDCSR": "feddcsr",
    }
    return aliases.get(name, name.lower())


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sources", nargs="+", default=None,
                        help="源域列表，默认 CPU 下用小三域；--sources all 跑全部")
    parser.add_argument("--targets", nargs="+", default=ALL_DOMAINS,
                        help="目标域列表，默认全部四个 Amazon 域")
    parser.add_argument("--methods", nargs="+", default=["no_transfer", "transfr", "fedbasa"])
    parser.add_argument("--seeds", type=int, nargs="+", default=[1])
    parser.add_argument("--output_root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--cpu", action="store_true", help="强制用 CPU（本地无 GPU 时使用）")
    parser.add_argument("--force_gpu", action="store_true")
    parser.add_argument("--dry_run", action="store_true", help="只打印命令，不实际运行")
    parser.add_argument("--max_runs", type=int, default=None, help="最多运行几次（调试用）")
    args = parser.parse_args()
    args.methods = [normalize_method_name(m) for m in args.methods]
    unsupported = [m for m in args.methods if m in DEDICATED_ORIGINAL_METHODS]
    if unsupported:
        names = ", ".join(unsupported)
        raise SystemExit(
            f"{names} require their dedicated original-code reproduction path; "
            "this adapter orchestrator will not substitute proxy methods for original-name baselines."
        )
    unknown = [m for m in args.methods if m not in METHOD_ARGS]
    if unknown:
        raise SystemExit(f"Unknown methods: {', '.join(unknown)}")

    # 决定用 CPU 还是 GPU
    use_cpu = args.cpu or not args.force_gpu

    # 解析源域
    sources = args.sources
    if sources is None:
        sources = CPU_SOURCES if use_cpu else ALL_DOMAINS
    if "all" in sources:
        sources = ALL_DOMAINS

    targets = ["all"] if "all" in args.targets else args.targets
    if "all" in targets:
        targets = ALL_DOMAINS

    args.output_root.mkdir(parents=True, exist_ok=True)
    master: dict = {}
    run_count = 0

    for seed in args.seeds:
        master[seed] = {}

        # ── Phase 1: 源域联邦预训练 ──────────────────────────────────────────
        source_adapters: dict[str, Path | None] = {}
        for source in sources:
            if args.max_runs is not None and run_count >= args.max_runs:
                break
            print(f"\n{'='*60}\n[PRETRAIN] {source}  seed={seed}\n{'='*60}")
            adapter_path = pretrain_source(source, seed, args.output_root, use_cpu, args.dry_run)
            source_adapters[source] = adapter_path
            run_count += 1
            # 记录预训练结果
            run_dir = args.output_root / f"pretrain_{source}_seed{seed}"
            m = load_metrics(run_dir)
            if m:
                master[seed][f"pretrain_{source}"] = {"phase": "pretrain", "domain": source, **m}

        # ── Phase 2a: 从零基线 (scratch/fedprox/fedadam, 每个目标域只跑一次) ──
        scratch_methods = [m for m in args.methods if m in SCRATCH_METHODS]
        for sm in scratch_methods:
            for target in targets:
                if args.max_runs is not None and run_count >= args.max_runs:
                    break
                print(f"\n{'='*60}\n[{sm.upper()}] target={target}  seed={seed}\n{'='*60}")
                run_name = adapt_target(None, target, sm, seed,
                                        args.output_root, use_cpu, args.dry_run)
                run_count += 1
                if run_name:
                    m = load_metrics(args.output_root / run_name, sm)
                    if m:
                        master[seed][run_name] = {
                            "source": None, "target": target, "method": sm, "seed": seed, **m}

        # ── Phase 2b: 迁移适应 (source -> target, method ∈ {transfr, fedbasa}) ─
        transfer_methods = [m for m in args.methods if m in TRANSFER_METHODS]
        for source in sources:
            for target in targets:
                if source == target:
                    continue
                for method in transfer_methods:
                    if args.max_runs is not None and run_count >= args.max_runs:
                        break
                    print(f"\n{'='*60}\n[TRANSFER] {source} -> {target}  method={method}  seed={seed}\n{'='*60}")
                    run_name = adapt_target(
                        source, target, method, seed,
                        args.output_root, use_cpu, args.dry_run,
                        source_adapters.get(source)
                    )
                    run_count += 1
                    if run_name:
                        m = load_metrics(args.output_root / run_name, method)
                        if m:
                            master[seed][run_name] = {
                                "source": source, "target": target, "method": method, "seed": seed, **m}

    master_path = args.output_root / "master_results.json"
    # Merge with any existing master so separate seed invocations don't overwrite each other.
    if master_path.exists():
        try:
            existing = json.loads(master_path.read_text(encoding="utf-8"))
            for seed_key, runs in existing.items():
                master.setdefault(seed_key, {}).update(
                    {k: v for k, v in runs.items() if k not in master.get(seed_key, {})}
                )
        except json.JSONDecodeError:
            pass
    master_path.write_text(json.dumps(master, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nDone. Master results -> {master_path}  (runs: {run_count})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
