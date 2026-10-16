"""Motivation figure (Fig. 1): uniform vs reliability-aware adaptation strength."""
import matplotlib
matplotlib.use("Agg")
import numpy as np
import matplotlib.pyplot as plt

plt.rcParams.update({"font.size": 8.5, "font.family": "DejaVu Sans"})

fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.35), dpi=200)

# (a) evidence vs adaptation strength
ax = axes[0]
freq = np.logspace(0, 3.3, 200)
uniform = np.full_like(freq, 0.8)
q = 1.0 / np.sqrt(freq + 1.0)
q = (q - q.min()) / (q.max() - q.min())
aware = 0.8 * (1 - 0.75 * q ** 0.7)
ax.set_xscale("log")
ax.plot(freq, uniform, "--", color="#9AA0A6", lw=1.8, label="uniform adaptation")
ax.plot(freq, aware, "-", color="#D93025", lw=2.0, label="FedPASA (evidence-aware)")
ax.fill_between(freq, aware, uniform, color="#D93025", alpha=0.06)
ax.set_xlabel("item interaction frequency (log scale)")
ax.set_ylabel("behavioral adaptation strength")
ax.set_title("(a) one uniform rule vs. evidence-gated routing", fontsize=8.8)
ax.legend(frameon=False, fontsize=7.4, loc="upper right")
ax.spines[["top", "right"]].set_visible(False)

# (b) domain density vs anchor weight
ax = axes[1]
eta = np.linspace(0.2, 2.2, 200)
rho = 0.2 * np.clip((1.5 - eta) / (1.5 - 1.0), 0, 1)
ax.plot(eta, np.full_like(eta, 0.0), "--", color="#9AA0A6", lw=1.8, label="no anchoring")
ax.plot(eta, rho, "-", color="#1A73E8", lw=2.0, label="AutoAnchor $\\rho_{eff}$")
ax.axvspan(0.2, 1.0, color="#1A73E8", alpha=0.05)
ax.text(0.58, 0.155, "sparse target", fontsize=7.4, color="#1A73E8", ha="center")
ax.text(1.9, 0.155, "dense target", fontsize=7.4, color="#5f6368", ha="center")
ax.set_xlabel(r"target density $\eta_T=|\mathcal{U}_T|/|\mathcal{I}_T|$")
ax.set_ylabel("anchor weight")
ax.set_ylim(-0.01, 0.22)
ax.set_title("(b) density-gated semantic anchoring", fontsize=8.8)
ax.legend(frameon=False, fontsize=7.4, loc="center right")
ax.spines[["top", "right"]].set_visible(False)

fig.tight_layout(pad=0.6)
fig.savefig("../paper_www/figures/motivation.pdf")
print("saved figures/motivation.pdf")
