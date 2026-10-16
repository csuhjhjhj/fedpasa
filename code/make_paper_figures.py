"""Paper figures from www_v1 runs: t-SNE drift, sensitivity curves, bucket bars."""
from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path
from statistics import mean

import numpy as np
import torch
import torch.nn.functional as F

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "outputs" / "www_v1"
FIGDIR = ROOT.parent / "paper_www" / "figures"
sys.path.insert(0, str(ROOT))

plt.rcParams.update({"font.size": 8.5, "font.family": "DejaVu Sans"})


def tsne_figure():
    from run_transfr_style import (read_fedcia_dat, leave_one_out_split, make_clients,
                                   client_text_profile, item_coldness, create_adapter,
                                   item_representation)

    class Args:
        pass

    dat = ROOT / "data/amazon_v2_transfr20_30/movies_and_tv/movies_and_tv.dat"
    by_user = read_fedcia_dat(dat)
    train, valid, test = leave_one_out_split(by_user)
    max_item = max(max(v) for v in train.values())
    num_items = max_item + 1
    features = torch.load(ROOT / "features" / "movies.pt", map_location="cpu")[:num_items]
    features = F.normalize(features, dim=-1)
    cold = item_coldness(train, num_items)
    clients = make_clients(sorted(train), 5, 42, train, "iid", 1.0)
    clients = [c for c in clients if c]
    profiles = [client_text_profile(c, train, features) for c in clients]

    panels = {}
    for method, label in (("transfr", "TransFR (no anchor)"), ("fedpasa", "FedPASA (AutoAnchor)")):
        run = OUT / "figruns" / "music_to_movies" / method / "seed42"
        state = torch.load(run / "global_adapter.pt", map_location="cpu")
        args_dict = json.loads((run / "results.json").read_text(encoding="utf-8"))["args"]
        a = Args()
        for k, v in args_dict.items():
            setattr(a, k, v)
        a.num_items = num_items
        a.effective_semantic_anchor_weight = args_dict.get("effective_semantic_anchor_weight", 0.0)
        adapter = create_adapter(a, features.size(1), torch.device("cpu"))
        adapter.load_state_dict(state)
        adapter.eval()
        with torch.no_grad():
            cid = 0
            ids = torch.arange(num_items)
            adapted = item_representation(adapter, features, a, profiles[cid], cold, item_ids=ids)
        panels[label] = adapted

    # Joint projection: PCA (shared) -> t-SNE on concatenated [raw, transfr, fedpasa]
    from sklearn.decomposition import PCA
    from sklearn.manifold import TSNE

    rng = np.random.default_rng(0)
    # focus on a readable subset: 400 cold + 400 warm items
    counts = torch.zeros(num_items)
    for seq in train.values():
        for i in seq:
            counts[i] += 1
    cold_idx = torch.nonzero(counts <= 5).squeeze(-1).numpy()
    warm_idx = torch.nonzero(counts > 20).squeeze(-1).numpy()
    cold_sub = rng.choice(cold_idx, min(400, len(cold_idx)), replace=False)
    warm_sub = rng.choice(warm_idx, min(400, len(warm_idx)), replace=False)
    subset = np.concatenate([cold_sub, warm_sub])
    labels = np.array(["cold"] * len(cold_sub) + ["warm"] * len(warm_sub))

    labels_methods = list(panels)
    raw = features[subset].numpy()
    mats = [raw] + [panels[m][subset].numpy() for m in labels_methods]
    X = np.concatenate(mats, axis=0)
    pca = PCA(n_components=50, random_state=0).fit(X)
    Z = pca.transform(X)
    emb = TSNE(n_components=2, random_state=0, perplexity=30, init="pca").fit_transform(Z)

    n = len(subset)
    fig, axes = plt.subplots(1, 3, figsize=(7.0, 2.3), dpi=200, sharex=True, sharey=True)
    titles = ["Frozen prior $x_i$"] + labels_methods
    for ax_i, ax in enumerate(axes):
        seg = emb[ax_i * n:(ax_i + 1) * n]
        for lab, color, marker in (("cold", "#D93025", "o"), ("warm", "#1A73E8", "^")):
            m = labels == lab
            ax.scatter(seg[m, 0], seg[m, 1], s=4, alpha=0.5, c=color, marker=marker,
                       label=("cold ($\\leq$5)" if lab == "cold" else "warm ($>$20)") if ax_i == 0 else None)
        ax.set_title(titles[ax_i], fontsize=8)
        ax.set_xticks([]); ax.set_yticks([])
    axes[0].legend(frameon=False, fontsize=7, loc="upper right")
    fig.suptitle("Shared PCA$\\to$t-SNE projection of the same items before/after adaptation (Music$\\to$Movies, seed 42)", fontsize=8.5)
    fig.tight_layout(pad=0.5)
    fig.savefig(FIGDIR / "tsne_anchor.pdf")
    print("saved tsne_anchor.pdf")


def sensitivity_figure():
    base = OUT / "sensitivity"
    if not base.exists():
        print("no sensitivity runs yet")
        return
    sweeps = {"alpha": ("$\\alpha$ (routing strength)", [0.25, 0.4, 0.55, 0.7, 0.85]),
              "coldpower": ("$\\kappa$ (coldness power)", [0.5, 0.7, 1.0, 1.5]),
              "condscale": ("$\\lambda$ (conditioning)", [0.0, 0.2, 0.4, 0.6]),
              "anchorw": ("$\\rho$ (max anchor)", [0.0, 0.1, 0.2, 0.3, 0.5])}
    fig, axes = plt.subplots(1, 4, figsize=(7.0, 1.9), dpi=200, sharey=True)
    for ax, (name, (xlabel, values)) in zip(axes, sweeps.items()):
        xs, ys = [], []
        for v in values:
            pts = []
            for res in base.glob(f"{name}{v}/seed*/results.json"):
                m = json.loads(res.read_text(encoding="utf-8")).get("metrics_no_pap", {})
                if "s99_HR@20" in m:
                    pts.append(m["s99_HR@20"])
            if pts:
                xs.append(v); ys.append(mean(pts))
        ax.plot(xs, ys, "o-", color="#D93025", lw=1.6, ms=3.5)
        ax.set_xlabel(xlabel, fontsize=7.5)
        ax.tick_params(labelsize=7)
        ax.grid(alpha=0.25, lw=0.4)
        ax.spines[["top", "right"]].set_visible(False)
    axes[0].set_ylabel("HR@20")
    fig.suptitle("Sensitivity on Music$\\to$Movies (three seeds)", fontsize=8.5)
    fig.tight_layout(pad=0.5)
    fig.savefig(FIGDIR / "sensitivity.pdf")
    print("saved sensitivity.pdf")


def bucket_figure():
    cross = OUT / "cross"
    if not cross.exists():
        return
    buckets = ["zero", "1_5", "6_20", "20plus"]
    labels = ["0", "1-5", "6-20", ">20"]
    series = defaultdict(lambda: defaultdict(list))
    for res in cross.rglob("results.json"):
        parts = res.parent.relative_to(cross).parts
        if len(parts) != 3:
            continue
        method = parts[1]
        if method not in ("transfr", "fedmoe", "fedpasa"):
            continue
        d = json.loads(res.read_text(encoding="utf-8"))
        m = d.get("metrics_no_pap", {})
        for b in buckets:
            k = f"s99freq_{b}_HR@20"
            if k in m:
                series[method][b].append(m[k])
    fig, ax = plt.subplots(figsize=(3.4, 2.2), dpi=200)
    width = 0.26
    x = np.arange(len(buckets))
    for j, (method, name, color) in enumerate((("transfr", "TransFR", "#9AA0A6"),
                                               ("fedmoe", "FedMoE", "#1A73E8"),
                                               ("fedpasa", "FedPASA", "#D93025"))):
        ys = [mean(series[method][b]) if series[method][b] else 0 for b in buckets]
        ax.bar(x + (j - 1) * width, ys, width, label=name, color=color)
    ax.set_xticks(x); ax.set_xticklabels(labels)
    ax.set_xlabel("target item frequency bucket", fontsize=8)
    ax.set_ylabel("HR@20")
    ax.legend(frameon=False, fontsize=7.5)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout(pad=0.4)
    fig.savefig(FIGDIR / "buckets.pdf")
    print("saved buckets.pdf")


if __name__ == "__main__":
    which = sys.argv[1] if len(sys.argv) > 1 else "all"
    if which in ("tsne", "all"):
        tsne_figure()
    if which in ("sens", "all"):
        sensitivity_figure()
    if which in ("buckets", "all"):
        bucket_figure()
