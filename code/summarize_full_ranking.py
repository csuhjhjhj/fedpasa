"""Summarize stored full-catalog metrics from the WWW experiment ledger.

The cross-domain evaluator records full-ranking metrics and sampled-99 metrics
in each ``results.json``.  This script extracts the former without retraining.
"""
from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parent
RUN_ROOT = ROOT / "outputs" / "www_v1" / "cross"
OUTPUT = ROOT / "outputs" / "www_v1" / "derived" / "full_ranking_existing.json"

DIRECTIONS = [
    "music_to_books",
    "books_to_music",
    "music_to_movies",
    "movies_to_music",
    "books_to_movies",
    "movies_to_books",
]
METHODS = [
    "fcf",
    "scratch",
    "fedprox",
    "fedadam",
    "transfr",
    "l2sp",
    "fedper",
    "pfedme",
    "fedmoe",
    "fedpasa_base",
    "fedpasa",
]
METRICS = ["HR@10", "HR@20", "NDCG@10", "NDCG@20"]


def metrics_for(payload: dict, method: str) -> dict | None:
    if method == "fcf":
        return payload.get("metrics_fedmf")
    if method == "pfedme":
        return payload.get("metrics_ditto") or payload.get("metrics_no_pap")
    if method == "fedpasa_base":
        return payload.get("metrics_no_pap")
    if method == "fedpasa":
        # The personalized FedPASA result is a distinct trained checkpoint.
        # Never substitute the base-stage result when that checkpoint is absent.
        return payload.get("metrics_ditto")
    return payload.get("metrics_no_pap")


def collect() -> dict:
    raw: dict[str, dict[str, list[dict]]] = defaultdict(lambda: defaultdict(list))
    for direction in DIRECTIONS:
        for method in METHODS:
            directory = "fedpasa" if method == "fedpasa_base" else method
            for result_path in sorted((RUN_ROOT / direction / directory).glob("seed*/results.json")):
                payload = json.loads(result_path.read_text(encoding="utf-8"))
                metrics = metrics_for(payload, method)
                if not metrics or any(name not in metrics for name in METRICS):
                    continue
                raw[method][direction].append(
                    {"seed": int(payload["args"]["seed"]), **{name: float(metrics[name]) for name in METRICS}}
                )

    summary = {"raw": raw, "summary": {}}
    for method in METHODS:
        direction_rows = {}
        seed_macro = defaultdict(lambda: defaultdict(list))
        for direction in DIRECTIONS:
            rows = raw[method][direction]
            if not rows:
                continue
            direction_rows[direction] = {
                metric: {
                    "mean": float(np.mean([row[metric] for row in rows])),
                    "std": float(np.std([row[metric] for row in rows], ddof=1)),
                    "n": len(rows),
                }
                for metric in METRICS
            }
            for row in rows:
                for metric in METRICS:
                    seed_macro[row["seed"]][metric].append(row[metric])
        macro = {}
        for metric in METRICS:
            values = [np.mean(by_metric[metric]) for _, by_metric in sorted(seed_macro.items()) if by_metric[metric]]
            if values:
                macro[metric] = {
                    "mean": float(np.mean(values)),
                    "std": float(np.std(values, ddof=1)),
                    "n_seeds": len(values),
                    "seed_values": [float(value) for value in values],
                }
        summary["summary"][method] = {"directions": direction_rows, "macro": macro}
    return summary


def main() -> None:
    summary = collect()
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    for method in METHODS:
        macro = summary["summary"].get(method, {}).get("macro", {})
        if not macro:
            continue
        values = " ".join(
            f"{metric}={macro[metric]['mean']:.6f}+/-{macro[metric]['std']:.6f}"
            for metric in METRICS
        )
        print(f"{method:14s} {values}")
    print(OUTPUT)


if __name__ == "__main__":
    main()
