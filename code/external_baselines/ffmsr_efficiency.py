"""Efficiency statistics for the FFMSR baseline runs (no GPU required).

1. Wall-clock / GPU-train / eval times parsed from the run logs in ./logs.
2. Model parameter counts from the fine-tuned checkpoints (state_dict numel).
3. Federated-pretrain communication cost per round, derived from the official
   FedtrainTrainer code: each client uploads its full item-prompt matrix
   (n_items x 768 float32) and downloads the clustered prompt matrix of the
   same size once per pre-training epoch (round).
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / "FFMSR"))  # checkpoints pickle references to `model.*`
LOGS = HERE / "logs"
PAIRS_ROOT = HERE / "results" / "ffmsr"

N_ITEMS = {"Music": 10634, "Books": 89254, "Movies": 18643}  # conversion_summary.json
PLM_SIZE = 768
BYTES = 4

LOGS_OF_INTEREST = {
    "ffmsr_music_books_seed42_formal.log": ("music_books", 42, False),
    "ffmsr_books_movies_seed42_formal.log": ("books_movies", 42, False),
    "ffmsr_music_movies_seed123.log": ("music_movies", 123, True),
    "ffmsr_music_books_seed123.log": ("music_books", 123, True),
    "ffmsr_books_movies_seed123.log": ("books_movies", 123, True),
}

RE_TRAIN_EPOCH = re.compile(r"epoch (\d+) training \[time: ([\d.]+)s, (train loss_A|train loss_B|train loss)")
RE_EVAL_EPOCH = re.compile(r"epoch (\d+) evaluating \[time: ([\d.]+)s")
RE_STAMP = re.compile(r"^(\d{2} \w{3}) (\d{2}:\d{2})")


def parse_log(path: Path):
    fed_epoch_times = {}   # epoch -> time (A and B lines share the same elapsed value)
    ft_train_times = []
    ft_eval_times = []
    stamps = []
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        m = RE_TRAIN_EPOCH.search(line)
        if m:
            epoch, t, kind = int(m.group(1)), float(m.group(2)), m.group(3)
            if kind == "train loss":
                ft_train_times.append(t)
            else:
                fed_epoch_times[epoch] = t  # A/B lines carry identical elapsed time
        m = RE_EVAL_EPOCH.search(line)
        if m:
            ft_eval_times.append(float(m.group(2)))
        m = RE_STAMP.match(line)
        if m:
            stamps.append(m.group(2))
    return {
        "pretrain_train_time_s": sum(fed_epoch_times.values()),
        "pretrain_epochs": len(fed_epoch_times),
        "finetune_train_time_s": sum(ft_train_times),
        "finetune_epochs": len(ft_train_times),
        "eval_time_s": sum(ft_eval_times),
        "log_first_stamp": stamps[0] if stamps else None,
        "log_last_stamp": stamps[-1] if stamps else None,
    }


def param_stats(pair: str, seed: int):
    """Parameter counts from one fine-tuned checkpoint per domain."""
    pair_dir = PAIRS_ROOT / pair / f"seed{seed}"
    out = {}
    months = ("Jan", "Feb", "Mar", "Apr", "May", "Jun",
              "Jul", "Aug", "Sep", "Oct", "Nov", "Dec")
    for path in sorted(pair_dir.glob("MLTRec-*.pth")):
        if path.stem.split("-")[1] not in months:
            continue
        state = torch.load(path, map_location="cpu", weights_only=False)
        domain = state["config"]["dataset"]
        if domain in out:  # keep the latest file per domain
            continue
        sd = state["state_dict"]
        groups = {}
        for name, tensor in sd.items():
            top = name.split(".")[0]
            groups[top] = groups.get(top, 0) + tensor.numel()
        out[domain] = {
            "total_params": sum(t.numel() for t in sd.values()),
            "by_module": dict(sorted(groups.items(), key=lambda kv: -kv[1])),
            "checkpoint": path.name,
        }
    return out


def comm_stats():
    rounds = {"music_movies": 5, "music_books": 1, "books_movies": 1}
    pairs = {"music_movies": ("Music", "Movies"),
             "music_books": ("Music", "Books"),
             "books_movies": ("Books", "Movies")}
    out = {}
    for pair, (a, b) in pairs.items():
        per_client_up = {d: N_ITEMS[d] * PLM_SIZE * BYTES for d in (a, b)}
        round_total = 2 * sum(per_client_up.values())  # upload + download, both clients
        out[pair] = {
            "rounds": rounds[pair],
            "upload_per_client_per_round_MB": {d: round(v / 1e6, 1) for d, v in per_client_up.items()},
            "total_per_round_MB": round(round_total / 1e6, 1),
            "total_all_rounds_MB": round(round_total * rounds[pair] / 1e6, 1),
        }
    return out


def main():
    report = {"logs": {}, "params": {}, "communication": comm_stats(),
              "notes": [
                  "seed42 logs uncontended; seed123 logs recorded while other tenants "
                  "shared the GPU (2.3-149 s/it swings), so seed123 times are upper bounds.",
                  "music_movies seed42 has no console log; wall clock estimated from "
                  "checkpoint file mtimes (13:03 -> 13:51, ~48 min).",
                  "communication: official FedtrainTrainer exchanges the full item-prompt "
                  "matrix (n_items x 768 fp32) per client per pre-train round, up and down.",
              ]}
    for fname, (pair, seed, contended) in LOGS_OF_INTEREST.items():
        path = LOGS / fname
        if path.exists():
            stats = parse_log(path)
            stats["pair"] = pair
            stats["seed"] = seed
            stats["gpu_contended"] = contended
            report["logs"][fname] = stats
    for pair, seed in [("music_movies", 42), ("music_books", 42), ("books_movies", 42)]:
        report["params"][f"{pair}/seed{seed}"] = param_stats(pair, seed)
    out = HERE / "results" / "ffmsr" / "efficiency_stats.json"
    out.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
