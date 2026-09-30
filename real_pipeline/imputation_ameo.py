"""
Missing-data experiment on real AMEO 2015 data: does GAIN help?

Protocol: the always-observed numeric inputs (10th %, 12th %, college GPA,
English, Logical, Quant, five personality scores) are masked completely at
random (MCAR) and missing at random (MAR: low-10th-% students more likely to
miss GPA / 12th %) at 10 / 20 / 30 %. Each mask is imputed with mean, KNN (k=5),
MICE (IterativeImputer) and GAIN (Yoon et al., ICML 2018). We report
normalised RMSE on the masked entries and downstream macro-F1 of the proposed
RandomForest after rebuilding ASR/SSR/TSR from the imputed data. 3 repeats.

Run:  python real_pipeline/imputation_ameo.py
Out:  outputs/real/imputation_results.json, imputation_report.txt, fig_imputation.png
"""
import json
import os
import sys
import time
import warnings

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.ensemble import RandomForestClassifier
from sklearn.experimental import enable_iterative_imputer  # noqa: F401
from sklearn.impute import IterativeImputer, KNNImputer, SimpleImputer
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(__file__))
import ameo_features as af  # noqa: E402
from career_mapping import map_designation  # noqa: E402
from experiments_ameo import OUT, SEED, feature_sets, spec_group  # noqa: E402

COLS = ["10percentage", "12percentage", "collegeGPA", "English", "Logical", "Quant",
        "conscientiousness", "agreeableness", "extraversion", "nueroticism",
        "openess_to_experience"]
REPORT, RESULTS = [], []


def log(s=""):
    print(s)
    REPORT.append(s)


def gain_impute(X, iters=10000, batch=128, hint_rate=0.9, alpha=100.0, seed=SEED):
    """GAIN (Yoon, Jordon & van der Schaar, 2018) with the authors' default
    settings (alpha=100, hint rate 0.9, 10,000 iterations). X: array with NaNs."""
    torch.manual_seed(seed)
    r = np.random.default_rng(seed)
    n, dim = X.shape
    M = (~np.isnan(X)).astype(np.float32)
    sc = MinMaxScaler()
    Xs = np.nan_to_num(sc.fit_transform(X), nan=0.0).astype(np.float32)

    def mlp():
        return nn.Sequential(nn.Linear(dim * 2, dim * 2), nn.ReLU(),
                             nn.Linear(dim * 2, dim * 2), nn.ReLU(),
                             nn.Linear(dim * 2, dim), nn.Sigmoid())
    G, D = mlp(), mlp()
    oG, oD = torch.optim.Adam(G.parameters()), torch.optim.Adam(D.parameters())
    Xt, Mt = torch.tensor(Xs), torch.tensor(M)
    for _ in range(iters):
        i = r.integers(0, n, batch)
        x, m = Xt[i], Mt[i]
        z = torch.rand_like(x) * 0.01
        x = m * x + (1 - m) * z
        b = (torch.rand_like(m) < hint_rate).float()
        h = b * m + 0.5 * (1 - b)
        g = G(torch.cat([x, m], 1))
        xh = m * x + (1 - m) * g
        dp = D(torch.cat([xh.detach(), h], 1))
        lD = -torch.mean(m * torch.log(dp + 1e-8) + (1 - m) * torch.log(1 - dp + 1e-8))
        oD.zero_grad()
        lD.backward()
        oD.step()
        dp = D(torch.cat([xh, h], 1))
        lG = -torch.mean((1 - m) * torch.log(dp + 1e-8)) + \
            alpha * torch.mean((m * x - m * g) ** 2) / torch.mean(m)
        oG.zero_grad()
        lG.backward()
        oG.step()
    with torch.no_grad():
        z = torch.rand_like(Xt) * 0.01
        x = Mt * Xt + (1 - Mt) * z
        g = G(torch.cat([x, Mt], 1))
        out = (Mt * Xt + (1 - Mt) * g).numpy()
    return sc.inverse_transform(out)


IMPUTERS = {
    "Mean": lambda X: SimpleImputer(strategy="mean").fit_transform(X),
    "KNN (k=5)": lambda X: KNNImputer(n_neighbors=5).fit_transform(X),
    "MICE": lambda X: IterativeImputer(max_iter=15, random_state=SEED).fit_transform(X),
    "GAIN": gain_impute,
}


def make_mask(X, rate, mech, r):
    n, d = X.shape
    if mech == "MCAR":
        return r.random((n, d)) < rate
    # MAR: missingness in 12th % and GPA driven by (observed) 10th %; others MCAR
    miss = r.random((n, d)) < rate
    rank = pd.Series(X[:, 0]).rank(pct=True).values
    p = np.clip(rate * 2 * (1 - rank), 0, 0.95)
    for j in [1, 2]:
        miss[:, j] = r.random(n) < p
    miss[:, 0] = False
    return miss


def downstream(df_imp, tr, te, y):
    feats = af.build_features(df_imp)
    d = pd.concat([df_imp, feats], axis=1)
    X = feature_sets(d)["Proposed (Academic + Behavioral)"].values.astype(float)
    m = RandomForestClassifier(n_estimators=200, min_samples_leaf=3,
                               class_weight="balanced_subsample", n_jobs=-1,
                               random_state=SEED).fit(X[tr], y[tr])
    return f1_score(y[te], m.predict(X[te]), average="macro")


def main():
    df = af.clean(pd.read_excel(af.RAW))
    df["career_domain"] = df["Designation"].map(map_designation)
    df = df[df["career_domain"].notna()].reset_index(drop=True)
    df["spec_group"] = df["Specialization"].map(spec_group)
    y = df["career_domain"].values
    tr, te = train_test_split(np.arange(len(df)), test_size=0.2, stratify=y, random_state=SEED)
    X_true = df[COLS].values.astype(float)
    rng_ = X_true.max(0) - X_true.min(0)
    f1_full = downstream(df, tr, te, y)
    log(f"Complete-data downstream macro-F1 (reference): {f1_full:.4f}")
    for mech in ["MCAR", "MAR"]:
        for rate in [0.1, 0.2, 0.3]:
            for rep in range(3):
                r = np.random.default_rng(SEED + rep)
                miss = make_mask(X_true, rate, mech, r)
                X_miss = X_true.copy()
                X_miss[miss] = np.nan
                for name, fn in IMPUTERS.items():
                    t0 = time.perf_counter()
                    X_imp = fn(X_miss)
                    secs = time.perf_counter() - t0
                    nrmse = float(np.sqrt(np.mean(((X_imp[miss] - X_true[miss]) /
                                                   np.broadcast_to(rng_, X_true.shape)[miss]) ** 2)))
                    d_imp = df.copy()
                    d_imp[COLS] = X_imp
                    f1 = downstream(d_imp, tr, te, y)
                    RESULTS.append({"mechanism": mech, "rate": rate, "repeat": rep, "method": name,
                                    "nrmse": nrmse, "macro_f1": f1, "seconds": secs})
            sub = pd.DataFrame([x for x in RESULTS if x["mechanism"] == mech and x["rate"] == rate])
            agg = sub.groupby("method")[["nrmse", "macro_f1", "seconds"]].agg(["mean", "std"])
            log(f"\n{mech} {int(rate * 100)}% missing (3 repeats):")
            for mth in IMPUTERS:
                a = agg.loc[mth]
                log(f"   {mth:<10s} NRMSE={a[('nrmse', 'mean')]:.4f}+/-{a[('nrmse', 'std')]:.4f}  "
                    f"macro-F1={a[('macro_f1', 'mean')]:.4f}+/-{a[('macro_f1', 'std')]:.4f}  "
                    f"time={a[('seconds', 'mean')]:.1f}s")

    res = pd.DataFrame(RESULTS)
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.6))
    for ax, mech in zip(axes, ["MCAR", "MAR"]):
        for mth, mk in zip(IMPUTERS, ["o", "s", "^", "D"]):
            g = res[(res.mechanism == mech) & (res.method == mth)].groupby("rate")["nrmse"]
            ax.errorbar(g.mean().index * 100, g.mean(), yerr=g.std(), marker=mk, capsize=3, label=mth)
        ax.set_title(mech)
        ax.set_xlabel("Missing rate (%)")
        ax.set_ylabel("Normalised RMSE")
        ax.grid(alpha=0.3)
    axes[0].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "fig_imputation.png"), dpi=300)
    plt.close(fig)
    with open(os.path.join(OUT, "imputation_results.json"), "w") as fh:
        json.dump({"complete_data_macro_f1": f1_full, "runs": RESULTS}, fh, indent=2)
    with open(os.path.join(OUT, "imputation_report.txt"), "w", encoding="utf-8") as fh:
        fh.write("\n".join(REPORT))


if __name__ == "__main__":
    main()
