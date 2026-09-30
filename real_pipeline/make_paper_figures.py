"""
Re-draws the paper figures at IEEE column width (3.5 in) with legible 7-pt text,
from the saved results (confusion matrix, threshold sweep, importance) and a
re-computed K-Means solution (identical settings to experiments_ameo.py).

Run:  python real_pipeline/make_paper_figures.py
Out:  outputs/real/paper_fig_*.png
"""
import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, os.path.dirname(__file__))
from experiments_ameo import BEH_CORE, OUT, SEED, load  # noqa: E402

plt.rcParams.update({"font.family": "serif", "font.serif": ["Times New Roman"], "font.size": 7,
                     "axes.titlesize": 7, "axes.labelsize": 7, "xtick.labelsize": 6.5,
                     "ytick.labelsize": 6.5, "legend.fontsize": 6.5})
W = 3.45
A = json.load(open(os.path.join(OUT, "ameo_results.json")))

# confusion matrix
cm = np.array(A["test_confusion_matrix"]["matrix"])
short = ["Software", "QA /\nTesting", "IT infra /\nSupport", "Data /\nBusiness", "Core\nEngg.", "Mgmt /\nSales"]
fig, ax = plt.subplots(figsize=(W, 2.9))
rates = cm / cm.sum(axis=1, keepdims=True)
im = ax.imshow(rates, cmap="Blues", vmin=0, vmax=1)
for i in range(6):
    for j in range(6):
        ax.text(j, i, cm[i, j], ha="center", va="center", fontsize=6.5,
                color="white" if rates[i, j] > 0.5 else "black")
ax.set_xticks(range(6), short, fontsize=6)
ax.set_yticks(range(6), short, fontsize=6)
ax.set_xlabel("Predicted domain")
ax.set_ylabel("Actual domain")
cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03)
cb.set_label("Row-normalized rate")
fig.tight_layout(pad=0.3)
fig.savefig(os.path.join(OUT, "paper_fig_confusion.png"), dpi=400)
plt.close(fig)

# threshold sweep
sw = pd.DataFrame(A["validation_layer"]["test_sweep"])
th = A["validation_layer"]["threshold"]
fig, ax = plt.subplots(figsize=(W, 2.2))
ax.plot(sw["threshold"], sw["accepted_acc"], "o-", ms=3, label="Accuracy of approved")
ax.plot(sw["threshold"], sw["accepted_macro_f1"], "s-", ms=3, label="Macro-F1 of approved")
ax.plot(sw["threshold"], sw["coverage"], "^--", ms=3, color="gray", label="Coverage")
ax.axvline(th, color="red", ls=":", lw=1, label=f"Calibrated θ = {th:.2f}")
ax.set_xlabel("Domain-SSR threshold θ")
ax.set_ylabel("Rate")
ax.grid(alpha=0.3)
ax.legend(loc="center right", frameon=True)
fig.tight_layout(pad=0.3)
fig.savefig(os.path.join(OUT, "paper_fig_threshold.png"), dpi=400)
plt.close(fig)

# permutation importance
imp = pd.DataFrame(A["permutation_importance"]).head(12).iloc[::-1]
nice = {"10percentage": "10th-grade %", "12percentage": "12th-grade %", "collegeGPA": "College GPA",
        "CollegeTier": "College tier"}
labels = [nice.get(f, f.replace("ssr_", "SSR: ").replace("spec_", "Spec.: ")
                   .replace("ComputerProgramming", "Programming")
                   .replace("ElectronicsAndSemicon", "Electronics")) for f in imp["feature"]]
colors = ["#c0392b" if f in BEH_CORE else ("#e67e22" if f.startswith("ssr_") else "#7f8c8d")
          for f in imp["feature"]]
fig, ax = plt.subplots(figsize=(W, 2.6))
ax.barh(labels, imp["mean"], xerr=imp["sd"], color=colors, error_kw={"lw": 0.7})
ax.set_xlabel("Drop in test macro-F1 when feature is permuted")
ax.axvline(0, color="black", lw=0.6)
from matplotlib.patches import Patch  # noqa: E402
ax.legend(handles=[Patch(color="#c0392b", label="ASR / SSR / TSR"),
                   Patch(color="#e67e22", label="Domain SSR components"),
                   Patch(color="#7f8c8d", label="Academic record")], loc="lower right")
fig.tight_layout(pad=0.3)
fig.savefig(os.path.join(OUT, "paper_fig_importance.png"), dpi=400)
plt.close(fig)

# behavioral profiles (same K-Means settings as experiments_ameo.run_clustering)
df = load()
Z = StandardScaler().fit_transform(df[BEH_CORE].values)
km = KMeans(3, n_init=20, random_state=SEED).fit(Z)
cent = pd.DataFrame(km.cluster_centers_, columns=BEH_CORE)
best = df.groupby(km.labels_)["CSS"].mean().idxmax()
lab_map = {"ASR": "Academic-shock", "SSR": "Skill-deficit", "TSR": "Time-delayed"}
names = {c: ("Resilient" if c == best else lab_map[cent.loc[c].idxmin()]) for c in cent.index}
risk = pd.Series(km.labels_).map(names).values
jit = np.random.default_rng(SEED).normal(0, 0.02, len(df))
fig, ax = plt.subplots(figsize=(W, 2.4))
for r, col in [("Resilient", "#27ae60"), ("Skill-deficit", "#f39c12"), ("Time-delayed", "#c0392b")]:
    m = risk == r
    ax.scatter(df.loc[m, "SSR"], df.loc[m, "TSR"] + jit[m], s=2, alpha=0.35, c=col,
               label=f"{r} (n={m.sum()})", rasterized=True)
ax.set_xlabel("Skill Shock Response (SSR)")
ax.set_ylabel("Time Shock Response (TSR, jittered)")
ax.legend(loc="lower left", markerscale=4, frameon=True)
fig.tight_layout(pad=0.3)
fig.savefig(os.path.join(OUT, "paper_fig_profiles.png"), dpi=400)
plt.close(fig)
print("saved paper figures")
