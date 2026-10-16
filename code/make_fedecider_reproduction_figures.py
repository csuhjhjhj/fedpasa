"""Create PNG-only figures from the fresh FeDecider-style reproduction."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.sans-serif"] = ["Arial", "DejaVu Sans", "Liberation Sans"]
plt.rcParams["svg.fonttype"] = "none"
plt.rcParams.update({
    "font.size": 10,
    "axes.spines.right": False,
    "axes.spines.top": False,
    "axes.linewidth": 0.8,
    "legend.frameon": False,
})

ROOT = Path(__file__).resolve().parent
DIRECTIONS = [
    "music_to_books", "music_to_movies", "books_to_movies",
    "books_to_music", "movies_to_books", "movies_to_music",
]
LABELS = ["M→B", "M→V", "B→V", "B→M", "V→B", "V→M"]
SEEDS = [42, 123, 456]
BLUE = "#0F4D92"
BLUE2 = "#3775BA"
RED = "#B64342"
TEAL = "#42949E"
GRAY = "#767676"
LIGHT = "#CFCECE"


def load_runs(base: Path) -> dict[tuple[str, int], dict]:
    runs = {}
    for direction in DIRECTIONS:
        for seed in SEEDS:
            path = base / direction / "fedecider" / f"seed{seed}" / "results.json"
            if not path.exists():
                raise FileNotFoundError(path)
            runs[(direction, seed)] = json.loads(path.read_text(encoding="utf-8"))
    return runs


def metric_matrix(runs: dict, key: str) -> np.ndarray:
    return np.array([
        [runs[(direction, seed)]["metrics_no_pap"][key] for seed in SEEDS]
        for direction in DIRECTIONS
    ], dtype=float)


def save(fig: plt.Figure, out: Path) -> None:
    fig.tight_layout(pad=1.2)
    fig.savefig(out, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def bar_with_sd(values: np.ndarray, ylabel: str, title: str, out: Path,
                color: str = BLUE, decimals: int = 4) -> None:
    mean = values.mean(axis=1)
    sd = values.std(axis=1, ddof=1)
    fig, ax = plt.subplots(figsize=(6.6, 4.0))
    x = np.arange(len(LABELS))
    bars = ax.bar(x, mean, yerr=sd, capsize=4, color=color,
                  edgecolor="black", linewidth=0.6)
    ax.set_xticks(x, LABELS)
    ax.set_ylabel(ylabel)
    ax.set_title(title, fontweight="bold")
    ax.grid(axis="y", color="#E6E6E6", linewidth=0.7)
    pad = max(mean.max() * 0.025, 1e-5)
    for b, v, e in zip(bars, mean, sd):
        ax.text(b.get_x() + b.get_width() / 2, v + e + pad,
                f"{v:.{decimals}f}", ha="center", va="bottom", fontsize=8)
    ax.set_ylim(0, max(mean + sd) * 1.18)
    fig.text(0.5, 0.005, "Mean ± SD across 3 seeds", ha="center", fontsize=8, color=GRAY)
    save(fig, out)


def seed_points(values: np.ndarray, ylabel: str, title: str, out: Path) -> None:
    fig, ax = plt.subplots(figsize=(6.6, 4.0))
    x = np.arange(len(LABELS))
    markers = ["o", "s", "^"]
    colors = [BLUE, TEAL, RED]
    for j, seed in enumerate(SEEDS):
        ax.scatter(x + (j - 1) * 0.08, values[:, j], s=42, marker=markers[j],
                   color=colors[j], edgecolor="white", linewidth=0.5, label=f"Seed {seed}", zorder=3)
    ax.scatter(x, values.mean(axis=1), color="#272727", marker="D", s=34,
               label="Mean", zorder=2)
    ax.set_xticks(x, LABELS)
    ax.set_ylabel(ylabel)
    ax.set_title(title, fontweight="bold")
    ax.grid(axis="y", color="#E6E6E6", linewidth=0.7)
    ax.legend(ncol=4, loc="upper center", bbox_to_anchor=(0.5, -0.14))
    save(fig, out)


def compare_bars(old: np.ndarray, new: np.ndarray, ylabel: str, title: str, out: Path) -> None:
    old_mean, new_mean = old.mean(axis=1), new.mean(axis=1)
    x = np.arange(len(LABELS)); width = 0.36
    fig, ax = plt.subplots(figsize=(6.6, 4.0))
    ax.bar(x - width / 2, old_mean, width, color=LIGHT, edgecolor="black", linewidth=0.5,
           label="Historical table run")
    ax.bar(x + width / 2, new_mean, width, color=BLUE, edgecolor="black", linewidth=0.5,
           label="Fresh reproduction")
    ax.set_xticks(x, LABELS)
    ax.set_ylabel(ylabel)
    ax.set_title(title, fontweight="bold")
    ax.grid(axis="y", color="#E6E6E6", linewidth=0.7)
    ax.legend()
    save(fig, out)


def delta_bars(old: np.ndarray, new: np.ndarray, ylabel: str, title: str, out: Path) -> None:
    delta = new.mean(axis=1) - old.mean(axis=1)
    colors = [BLUE if v >= 0 else RED for v in delta]
    fig, ax = plt.subplots(figsize=(6.6, 4.0))
    x = np.arange(len(LABELS))
    bars = ax.bar(x, delta, color=colors, edgecolor="black", linewidth=0.5)
    ax.axhline(0, color="#272727", linewidth=0.9)
    ax.set_xticks(x, LABELS)
    ax.set_ylabel(ylabel)
    ax.set_title(title, fontweight="bold")
    ax.grid(axis="y", color="#E6E6E6", linewidth=0.7)
    span = max(np.abs(delta).max(), 1e-4)
    pad = span * 0.045
    for b, v in zip(bars, delta):
        label = "0" if np.isclose(v, 0.0, atol=1e-10) else f"{v:+.4f}"
        ax.text(b.get_x() + b.get_width() / 2, v + (pad if v >= 0 else -pad),
                label, ha="center", va="bottom" if v >= 0 else "top", fontsize=8)
    ax.set_ylim(min(delta.min() - 3 * pad, -0.15 * span), max(delta.max() + 3 * pad, 0.15 * span))
    save(fig, out)


def resource_bar(values: np.ndarray, ylabel: str, title: str, out: Path,
                 reducer: str = "mean") -> None:
    center = values.mean(axis=1) if reducer == "mean" else values.max(axis=1)
    spread = values.std(axis=1, ddof=1) if reducer == "mean" else None
    fig, ax = plt.subplots(figsize=(6.6, 4.0))
    x = np.arange(len(LABELS))
    bars = ax.bar(x, center, yerr=spread, capsize=4 if spread is not None else 0,
                  color=TEAL, edgecolor="black", linewidth=0.5)
    ax.set_xticks(x, LABELS)
    ax.set_ylabel(ylabel)
    ax.set_title(title, fontweight="bold")
    ax.grid(axis="y", color="#E6E6E6", linewidth=0.7)
    for b, v in zip(bars, center):
        ax.text(b.get_x() + b.get_width() / 2, v + center.max() * 0.02,
                f"{v:.1f}", ha="center", va="bottom", fontsize=8)
    ax.set_ylim(0, center.max() * 1.16)
    save(fig, out)


def asymmetry(values: np.ndarray, ylabel: str, title: str, out: Path) -> None:
    means = values.mean(axis=1)
    pairs = [(0, 3, "Music↔Books"), (1, 5, "Music↔Movies"), (2, 4, "Books↔Movies")]
    fig, ax = plt.subplots(figsize=(6.6, 4.0))
    y = np.arange(len(pairs))
    for yi, (a, b, _) in zip(y, pairs):
        ax.barh(yi + 0.15, means[a], height=0.24, color=BLUE,
                edgecolor="black", linewidth=0.5)
        ax.barh(yi - 0.15, means[b], height=0.24, color=RED,
                edgecolor="black", linewidth=0.5)
        ax.text(means[a] + means.max() * 0.018, yi + 0.15, LABELS[a],
                ha="left", va="center", fontsize=8)
        ax.text(means[b] + means.max() * 0.018, yi - 0.15, LABELS[b],
                ha="left", va="center", fontsize=8)
    ax.set_yticks(y, [p[2] for p in pairs])
    ax.set_xlabel(ylabel)
    ax.set_title(title, fontweight="bold")
    ax.grid(axis="x", color="#E6E6E6", linewidth=0.7)
    ax.set_xlim(0, means.max() * 1.18)
    save(fig, out)


def training_comparison(old_runs: dict, new_runs: dict, out: Path) -> None:
    key = ("books_to_movies", 42)
    old = [x["train_loss"] for x in old_runs[key]["history"]]
    new = [x["train_loss"] for x in new_runs[key]["history"]]
    rounds = np.arange(1, len(new) + 1)
    fig, ax = plt.subplots(figsize=(6.6, 4.0))
    ax.plot(rounds, old, color=GRAY, marker="o", linewidth=1.8,
            label="Historical silent fallback")
    ax.plot(rounds, new, color=BLUE, marker="o", linewidth=2.2,
            label="Fresh run with source adapter")
    ax.set_xlabel("Federated round")
    ax.set_ylabel("Training loss")
    ax.set_title("Books→Movies seed 42: corrected transfer trajectory", fontweight="bold")
    ax.grid(color="#E6E6E6", linewidth=0.7)
    ax.legend()
    save(fig, out)


def load_fedpasa(base: Path, key: str) -> np.ndarray:
    rows = []
    for direction in DIRECTIONS:
        vals = []
        for path in sorted((base / direction / "pasa_ditto").glob("seed*/results.json")):
            data = json.loads(path.read_text(encoding="utf-8"))
            metrics = data.get("metrics_ditto") or data.get("metrics_no_pap") or {}
            if key in metrics:
                vals.append(metrics[key])
        if not vals:
            raise FileNotFoundError(f"No FedPASA results for {direction}")
        rows.append(vals)
    return np.array(rows, dtype=float)


def method_compare(fedecider_hr: np.ndarray, fedecider_nd: np.ndarray,
                   fedpasa_hr: np.ndarray, fedpasa_nd: np.ndarray, out: Path) -> None:
    labels = LABELS + ["Mean"]
    fede = np.r_[fedecider_hr.mean(axis=1), fedecider_hr.mean()]
    pasa = np.r_[fedpasa_hr.mean(axis=1), fedpasa_hr.mean()]
    x = np.arange(len(labels)); width = 0.36
    fig, ax = plt.subplots(figsize=(7.0, 4.0))
    ax.bar(x - width / 2, fede, width, color=LIGHT, edgecolor="black", linewidth=0.5,
           label="FeDecider-style")
    ax.bar(x + width / 2, pasa, width, color=BLUE, edgecolor="black", linewidth=0.5,
           label="FedPASA")
    ax.set_xticks(x, labels)
    ax.set_ylabel("Sampled-99 HR@20")
    ax.set_title("Main-method comparison under the common protocol", fontweight="bold")
    ax.grid(axis="y", color="#E6E6E6", linewidth=0.7)
    ax.legend()
    save(fig, out)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--new-root", type=Path,
                        default=ROOT / "outputs/reproductions/fedecider_v7_20261014/v7/fedecider")
    parser.add_argument("--old-root", type=Path,
                        default=ROOT / "outputs/www_v1/v7/fedecider")
    parser.add_argument("--fedpasa-root", type=Path,
                        default=ROOT / "outputs/www_v1/v7/ditto")
    parser.add_argument("--output-dir", type=Path,
                        default=ROOT / "outputs/reproductions/fedecider_v7_20261014/figures")
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    new, old = load_runs(args.new_root), load_runs(args.old_root)
    new_hr = metric_matrix(new, "s99_HR@20")
    new_nd = metric_matrix(new, "s99_NDCG@20")
    new_full_hr = metric_matrix(new, "HR@20")
    new_full_nd = metric_matrix(new, "NDCG@20")
    old_hr = metric_matrix(old, "s99_HR@20")
    old_nd = metric_matrix(old, "s99_NDCG@20")

    bar_with_sd(new_hr, "Sampled-99 HR@20", "FeDecider-style transfer accuracy",
                args.output_dir / "fig01_sampled_hr_by_direction.png")
    bar_with_sd(new_nd, "Sampled-99 NDCG@20", "FeDecider-style ranking quality",
                args.output_dir / "fig02_sampled_ndcg_by_direction.png", color=TEAL)
    bar_with_sd(new_full_hr, "Full-ranking HR@20", "Full-ranking transfer accuracy",
                args.output_dir / "fig03_full_hr_by_direction.png", decimals=5)
    bar_with_sd(new_full_nd, "Full-ranking NDCG@20", "Full-ranking ranking quality",
                args.output_dir / "fig04_full_ndcg_by_direction.png", color=TEAL, decimals=5)
    seed_points(new_hr, "Sampled-99 HR@20", "Seed-level HR@20 reproducibility",
                args.output_dir / "fig05_sampled_hr_seed_stability.png")
    seed_points(new_nd, "Sampled-99 NDCG@20", "Seed-level NDCG@20 reproducibility",
                args.output_dir / "fig06_sampled_ndcg_seed_stability.png")
    compare_bars(old_hr, new_hr, "Sampled-99 HR@20", "Historical vs fresh reproduction",
                 args.output_dir / "fig07_old_vs_new_hr.png")
    compare_bars(old_nd, new_nd, "Sampled-99 NDCG@20", "Historical vs fresh reproduction",
                 args.output_dir / "fig08_old_vs_new_ndcg.png")
    delta_bars(old_hr, new_hr, "Fresh − historical HR@20", "Reproduction delta by direction",
               args.output_dir / "fig09_reproduction_delta_hr.png")
    delta_bars(old_nd, new_nd, "Fresh − historical NDCG@20", "Reproduction delta by direction",
               args.output_dir / "fig10_reproduction_delta_ndcg.png")

    runtime = np.array([[new[(d, s)]["runtime"]["total_seconds"] for s in SEEDS] for d in DIRECTIONS])
    memory = np.array([[new[(d, s)]["runtime"]["cuda_max_memory_allocated_mb"] for s in SEEDS] for d in DIRECTIONS])
    resource_bar(runtime, "Runtime per run (s)", "Runtime by transfer direction",
                 args.output_dir / "fig11_runtime_by_direction.png")
    resource_bar(memory, "Peak CUDA allocation (MB)", "GPU memory by transfer direction",
                 args.output_dir / "fig12_gpu_memory_by_direction.png", reducer="max")
    asymmetry(new_hr, "Sampled-99 HR@20", "Directional transfer asymmetry",
              args.output_dir / "fig13_transfer_asymmetry_hr.png")
    asymmetry(new_nd, "Sampled-99 NDCG@20", "Directional ranking asymmetry",
              args.output_dir / "fig14_transfer_asymmetry_ndcg.png")
    training_comparison(old, new, args.output_dir / "fig15_corrected_training_trajectory.png")

    pasa_hr = load_fedpasa(args.fedpasa_root, "s99_HR@20")
    pasa_nd = load_fedpasa(args.fedpasa_root, "s99_NDCG@20")
    method_compare(new_hr, new_nd, pasa_hr, pasa_nd,
                   args.output_dir / "fig16_fedpasa_vs_fedecider.png")
    print(f"Wrote {len(list(args.output_dir.glob('*.png')))} PNG figures to {args.output_dir}")


if __name__ == "__main__":
    main()
