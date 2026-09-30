"""Block diagram of the revised pipeline at IEEE column width (Fig. 1)."""
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                   "outputs", "real", "fig_pipeline.png")
EDGE = "#34495e"
plt.rcParams.update({"font.family": "serif", "font.serif": ["Times New Roman"]})

fig, ax = plt.subplots(figsize=(3.45, 4.3))
ax.set_xlim(0, 100)
ax.set_ylim(0, 125)
ax.axis("off")


def box(x, y, w, h, text, fc, fs=6.2, bold=False, ls="-"):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.4,rounding_size=1.5",
                                fc=fc, ec=EDGE, lw=0.7, ls=ls))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs,
            fontweight="bold" if bold else "normal", linespacing=1.15)


def arrow(x1, y1, x2, y2, ls="-"):
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>", mutation_scale=7,
                                 lw=0.7, color=EDGE, ls=ls))


box(1, 108, 47, 15, "Academic trajectory\n10th % → 12th % → GPA\n+ completion years", "#d6eaf8")
box(52, 108, 47, 15, "Skill evidence\nAMCAT skill modules\n+ aptitude scores", "#d6eaf8")
box(1, 94, 98, 8, "Board / tier z-scoring, Eq. (1)  +  MICE imputation", "#d5f5e3")
arrow(24.5, 108, 24.5, 102.6)
arrow(75.5, 108, 75.5, 102.6)
box(1, 74, 31, 14, "ASR\nEqs. (2)–(3)\nshock + recovery", "#fadbd8")
box(34.5, 74, 31, 14, "TSR\nEqs. (4)–(5)\ntime lost", "#fadbd8")
box(68, 74, 31, 14, "SSR\nEqs. (6)–(7)\nskill beyond grades", "#fadbd8")
for x in [16.5, 50, 83.5]:
    arrow(x, 94, x, 88.6)
ax.plot([4, 96], [69, 69], color=EDGE, lw=0.9)
for x in [16.5, 50, 83.5]:
    ax.plot([x, x], [74, 69], color=EDGE, lw=0.7)
box(1, 50, 47, 13, "Random Forest\ncareer domain (6 classes)\nacademic + ASR/TSR/SSR", "#e8daef")
box(52, 50, 47, 13, "K-Means (k = 3)\nbehavioral profiles\n+ CSS summary, Eq. (8)", "#e8daef")
arrow(24.5, 69, 24.5, 63.6)
arrow(75.5, 69, 75.5, 63.6)
box(1, 29, 47, 14, "Domain-consistency check\nSSR(d) ≥ θ, Eq. (9)\nθ tuned on training folds", "#e8daef")
box(62, 29, 37, 14, "Outcome validation\nsalary, early exit\n(not for training)", "#eaeded", ls="--")
arrow(24.5, 50, 24.5, 43.6)
arrow(80.5, 50, 80.5, 43.6, ls="--")
box(1, 3, 98, 18, "Output: career domain + approved /\nlow-confidence flag + behavioral profile\n+ explanation",
    "#fcf3cf", fs=6.4, bold=True)
arrow(24.5, 29, 24.5, 21.6)
arrow(56, 50, 56, 21.6)
fig.tight_layout(pad=0.1)
fig.savefig(OUT, dpi=400)
print(OUT)
