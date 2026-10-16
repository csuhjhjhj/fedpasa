"""Source-Reliability Gate analysis figure + data for the paper."""
import json
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams.update({"font.size": 8.5, "font.family": "DejaVu Sans"})

# zero-shot valid HR (seed42) vs source benefit (src - nosrc, 5-seed mean)
pts = [
    ("music_to_movies", 0.1922, -0.0565),
    ("books_to_movies", 0.2062, +0.0114),
    ("music_to_books", 0.1970, -0.0126),
    ("books_to_music", 0.1934, -0.0134),
    ("movies_to_books", 0.2022, -0.0045),
    ("movies_to_music", 0.2028, +0.0026),
]
TAU = 0.2025

fig, ax = plt.subplots(figsize=(3.6, 2.6), dpi=200)
ax.axvspan(TAU, 0.215, color="#1A73E8", alpha=0.06)
ax.axhline(0, color="#9AA0A6", lw=0.8)
ax.axvline(TAU, color="#1A73E8", lw=1.0, ls="--")
offsets = {"music_to_movies": (4, 4), "books_to_movies": (5, 2), "music_to_books": (2, 4),
           "books_to_music": (4, 2), "movies_to_books": (2, -11), "movies_to_music": (6, 0)}
for name, x, y in pts:
    c = "#1A73E8" if y > 0 else "#D93025"
    ax.scatter(x, y * 100, s=28, color=c, zorder=3)
    lab = name.replace("_to_", "$\\to$")
    off = offsets[name]
    ha = "right" if off[0] < 0 else "left"
    ax.annotate(lab, (x, y * 100), textcoords="offset points", xytext=off, fontsize=6.2, ha=ha)
ax.text(TAU - 0.0018, -5.6, "accept source\n($\\tau=0.2025$)", fontsize=6.6, color="#1A73E8", ha="left")
ax.set_xlabel("source adapter zero-shot HR@20 on target validation", fontsize=7.6)
ax.set_ylabel("source benefit (src $-$ no-src), $\\times 10^{-2}$", fontsize=7.6)
ax.set_ylim(-6.5, 2.0)
ax.spines[["top", "right"]].set_visible(False)
fig.tight_layout(pad=0.4)
fig.savefig("../paper_www/figures/source_gate.pdf")

json.dump({"tau": TAU, "points": pts}, open("outputs/www_v1/source_gate.json", "w"))
print("saved figures/source_gate.pdf")
