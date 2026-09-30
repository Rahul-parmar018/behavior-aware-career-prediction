"""
Second real dataset: does ASR/SSR/TSR generalise beyond AMEO?

Source: UCI ML Repository #697 "Predict Students' Dropout and Academic Success"
(4,424 students, Polytechnic Institute of Portalegre; Realinho et al., Data 2022,
doi:10.3390/data7110146, CC BY 4.0).

Academic trajectory: previous-qualification grade -> admission grade ->
semester-1 grade -> semester-2 grade (4 time points, z-scored within course).
Task: academic survivability, Dropout vs Graduate (binary); "Enrolled" is
excluded because the outcome is not yet known.

Run:  python real_pipeline/experiments_uci.py
Out:  outputs/real/uci_results.json, outputs/real/uci_report.txt, fig_uci_*.png
"""
import json
import os
import sys
import warnings

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.model_selection import (RepeatedStratifiedKFold, cross_val_score,
                                     train_test_split)
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(__file__))
from experiments_ameo import (OUT, ROOT, SEED, bootstrap_ci, cohens_d, boot_d,  # noqa: E402
                              corrected_ttest, mcnemar, metrics)

RAW = os.path.join(ROOT, "data", "real", "uci_dropout.csv")
TAU, LAM = 0.5, 1.0
REPORT, RESULTS = [], {}


def log(s=""):
    print(s)
    REPORT.append(s)


def z_within(v, g):
    mu = v.groupby(g).transform("mean")
    sd = v.groupby(g).transform("std").replace(0, np.nan)
    return ((v - mu) / sd).fillna(0.0)


ZERO_FLOOR = -3.0


def z_semester(grade, course):
    """A semester grade of 0 means no unit was approved (the grade is undefined),
    not a real mark. Standardise only the defined grades, and give every
    no-approval semester the same floor so two failed semesters are not read as
    a 'recovery'."""
    defined = grade > 0
    z = pd.Series(ZERO_FLOOR, index=grade.index)
    z[defined] = z_within(grade[defined], course[defined])
    return z


def build(df, tau=TAU, lam=LAM):
    c = "Course"
    f = pd.DataFrame(index=df.index)
    zp = z_within(df["Previous qualification (grade)"], df[c])
    za = z_within(df["Admission grade"], df[c])
    z1 = z_semester(df["Curricular units 1st sem (grade)"], df[c])
    z2 = z_semester(df["Curricular units 2nd sem (grade)"], df[c])
    # ASR: transitions school->admission, admission->sem1 (entry shock), sem1->sem2
    d0, d1, d2 = za - zp, z1 - za, z2 - z1
    shock = (d1 < -tau).astype(float)          # setback on entering university
    f["academic_shock"] = shock
    f["ASR_raw"] = (d0 + d1 + d2) / 3 + lam * shock * d2
    f["ASR"] = f["ASR_raw"].rank(pct=True)
    # SSR: skill conversion - approved units per attempted assessment
    conv = []
    for s in ["1st", "2nd"]:
        ev = df[f"Curricular units {s} sem (evaluations)"]
        ap = df[f"Curricular units {s} sem (approved)"]
        conv.append((ap / ev.where(ev > 0)).fillna(0).clip(0, 1))
    f["SSR_raw"] = (conv[0] + conv[1]) / 2
    f["SSR"] = f["SSR_raw"].rank(pct=True)
    # TSR: time - late entry and postponed (not assessed) units
    delay = (df["Age at enrollment"] - 18).clip(lower=0, upper=10) / 10
    enr = df["Curricular units 1st sem (enrolled)"] + df["Curricular units 2nd sem (enrolled)"]
    wo = df["Curricular units 1st sem (without evaluations)"] + \
        df["Curricular units 2nd sem (without evaluations)"]
    postponed = (wo / enr.where(enr > 0)).fillna(1.0).clip(0, 1)
    f["TSR"] = 1 - (delay + postponed) / 2
    f["CSS"] = (f["ASR"] + f["SSR"] + f["TSR"]) / 3
    return f


def main():
    df = pd.read_csv(RAW, sep=";", encoding="utf-8-sig")
    df.columns = [col.strip() for col in df.columns]
    df = df[df["Target"] != "Enrolled"].reset_index(drop=True)
    y = (df["Target"] == "Dropout").astype(int).values
    feats = build(df)
    log(f"UCI #697, Dropout vs Graduate: n={len(df)}, dropout rate={y.mean():.1%}")
    log(f"Academic shock rate: {feats['academic_shock'].mean():.1%}")
    log("Correlations: \n" + feats[["ASR", "SSR", "TSR", "CSS"]].corr().round(3).to_string())

    course = pd.get_dummies(df["Course"], prefix="course", dtype=float)
    acad = pd.concat([df[["Previous qualification (grade)", "Admission grade",
                          "Curricular units 1st sem (grade)", "Curricular units 2nd sem (grade)",
                          "Daytime/evening attendance"]], course], axis=1)
    beh = feats[["ASR", "SSR", "TSR"]]
    socio = df[["Tuition fees up to date", "Debtor", "Scholarship holder", "Gender",
                "Displaced", "Unemployment rate", "Inflation rate", "GDP"]]
    fsets = {"Academic (grades + course)": acad,
             "Behavioral core (ASR, SSR, TSR)": beh,
             "Proposed (Academic + Behavioral)": pd.concat([acad, beh], axis=1),
             "Proposed + CSS (redundancy check)": pd.concat([acad, beh, feats[["CSS"]]], axis=1),
             "Proposed + socio-economic (extended)": pd.concat([acad, beh, socio], axis=1)}

    def rf():
        return RandomForestClassifier(n_estimators=400, min_samples_leaf=3,
                                      class_weight="balanced_subsample", n_jobs=-1,
                                      random_state=SEED)

    tr, te = train_test_split(np.arange(len(df)), test_size=0.2, stratify=y, random_state=SEED)
    log(f"Split: stratified 80/20 -> train {len(tr)}, test {len(te)}")
    rows, preds, probas = [], {}, {}
    for name, X in fsets.items():
        X = X.values.astype(float)
        for mname, m in {"RandomForest": rf(),
                         "LogisticRegression": make_pipeline(StandardScaler(), LogisticRegression(
                             max_iter=3000, class_weight="balanced"))}.items():
            m.fit(X[tr], y[tr])
            p = m.predict(X[te])
            pr = m.predict_proba(X[te])[:, 1]
            auc = roc_auc_score(y[te], pr)
            aucs = []
            r = np.random.default_rng(SEED)
            for _ in range(1000):
                i = r.integers(0, len(te), len(te))
                if len(np.unique(y[te][i])) == 2:
                    aucs.append(roc_auc_score(y[te][i], pr[i]))
            rows.append({"features": name, "model": mname, **metrics(y[te], p), "auc": auc,
                         "auc_ci": (float(np.percentile(aucs, 2.5)), float(np.percentile(aucs, 97.5))),
                         "ci": bootstrap_ci(y[te], p)})
            preds[(name, mname)], probas[(name, mname)] = p, pr
    log(f"\n{'Feature set':<40s}{'Model':<20s}{'Acc':>8s}{'Bal.Acc':>9s}{'MacroF1':>9s}"
        f"{'Kappa':>8s}{'AUC [95% CI]':>22s}")
    for r in rows:
        log(f"{r['features']:<40s}{r['model']:<20s}{r['accuracy']:8.3f}{r['balanced_accuracy']:9.3f}"
            f"{r['macro_f1']:9.3f}{r['kappa']:8.3f}   {r['auc']:.3f} [{r['auc_ci'][0]:.3f},{r['auc_ci'][1]:.3f}]")
    log("\nMcNemar, Academic vs Proposed:")
    mc = {}
    for mname in ["RandomForest", "LogisticRegression"]:
        chi2, p, n01, n10 = mcnemar(y[te], preds[("Academic (grades + course)", mname)],
                                    preds[("Proposed (Academic + Behavioral)", mname)])
        mc[mname] = {"chi2": chi2, "p": p, "n01": n01, "n10": n10}
        log(f"   {mname:<20s} chi2={chi2:.3f} p={p:.4g} (only academic correct={n01}, "
            f"only proposed correct={n10})")

    rcv = RepeatedStratifiedKFold(n_splits=5, n_repeats=5, random_state=SEED)
    cv = {}
    log("\nRepeated 5x5 CV, RandomForest, ROC-AUC:")
    for name in ["Academic (grades + course)", "Behavioral core (ASR, SSR, TSR)",
                 "Proposed (Academic + Behavioral)", "Proposed + CSS (redundancy check)"]:
        s = cross_val_score(rf(), fsets[name].values.astype(float), y, cv=rcv,
                            scoring="roc_auc", n_jobs=-1)
        cv[name] = s
        log(f"   {name:<40s} {s.mean():.4f} +/- {s.std(ddof=1):.4f}")
    n_te = len(df) // 5
    t, p = corrected_ttest(cv["Academic (grades + course)"], cv["Proposed (Academic + Behavioral)"],
                           len(df) - n_te, n_te)
    t2, p2 = corrected_ttest(cv["Proposed (Academic + Behavioral)"],
                             cv["Proposed + CSS (redundancy check)"], len(df) - n_te, n_te)
    log(f"   Corrected t-test Proposed vs Academic: t={t:.3f}, p={p:.4g}")
    log(f"   Corrected t-test Proposed+CSS vs Proposed: t={t2:.3f}, p={p2:.4g}")

    log("\nEach feature vs dropout (AUC, lower score -> dropout):")
    uni = {}
    for fname in ["ASR", "SSR", "TSR", "CSS"]:
        a = roc_auc_score(y, -feats[fname])
        uni[fname] = a
        log(f"   {fname}: AUC={a:.3f}")
    grad = feats.loc[y == 0, "CSS"].values
    drop = feats.loc[y == 1, "CSS"].values
    d = cohens_d(grad, drop)
    ci = boot_d(grad, drop)
    log(f"\nCohen's d, CSS graduates (n={len(grad)}, M={grad.mean():.3f}, SD={grad.std(ddof=1):.3f}) "
        f"vs dropouts (n={len(drop)}, M={drop.mean():.3f}, SD={drop.std(ddof=1):.3f}): "
        f"d={d:.3f} 95% CI [{ci[0]:.3f}, {ci[1]:.3f}]")
    shock_drop = y[feats["academic_shock"] == 1].mean()
    noshock_drop = y[feats["academic_shock"] == 0].mean()
    ct = pd.crosstab(feats["academic_shock"], y)
    chi2, p_s, _, _ = stats.chi2_contingency(ct)
    log(f"Dropout rate after an entry shock: {shock_drop:.1%} vs {noshock_drop:.1%} without "
        f"(chi2={chi2:.1f}, p={p_s:.3g})")
    # among shocked students, does the response (sem-2 recovery) matter?
    sh = feats["academic_shock"] == 1
    z2 = z_semester(df["Curricular units 2nd sem (grade)"], df["Course"])
    z1 = z_semester(df["Curricular units 1st sem (grade)"], df["Course"])
    rec = (z2 - z1)[sh] > 0
    log(f"Among shocked students: dropout {y[sh][rec].mean():.1%} if they recovered in sem 2 "
        f"vs {y[sh][~rec].mean():.1%} if not (n={rec.sum()} / {(~rec).sum()})")

    fig, ax = plt.subplots(figsize=(5, 4.2))
    for name, style in [("Academic (grades + course)", "--"),
                        ("Proposed (Academic + Behavioral)", "-"),
                        ("Behavioral core (ASR, SSR, TSR)", ":")]:
        fpr, tpr, _ = roc_curve(y[te], probas[(name, "RandomForest")])
        ax.plot(fpr, tpr, style, label=f"{name.split(' (')[0]} (AUC={roc_auc_score(y[te], probas[(name, 'RandomForest')]):.3f})")
    ax.plot([0, 1], [0, 1], color="gray", lw=0.8)
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "fig_uci_roc.png"), dpi=300)
    plt.close(fig)

    RESULTS.update({"n": len(df), "dropout_rate": float(y.mean()), "table": rows, "mcnemar": mc,
                    "cv_auc": {k: {"mean": float(v.mean()), "sd": float(v.std(ddof=1))}
                               for k, v in cv.items()},
                    "cv_ttest_proposed_vs_academic": {"t": t, "p": p},
                    "cv_ttest_css": {"t": t2, "p": p2}, "univariate_auc": uni,
                    "cohens_d_css": {"d": float(d), "ci": ci},
                    "shock_dropout": {"shock": float(shock_drop), "no_shock": float(noshock_drop),
                                      "chi2": float(chi2), "p": float(p_s)},
                    "recovery_among_shocked": {"recovered": float(y[sh][rec].mean()),
                                               "not_recovered": float(y[sh][~rec].mean())}})
    with open(os.path.join(OUT, "uci_results.json"), "w") as fh:
        json.dump(RESULTS, fh, indent=2, default=lambda o: o.item() if hasattr(o, "item") else str(o))
    with open(os.path.join(OUT, "uci_report.txt"), "w", encoding="utf-8") as fh:
        fh.write("\n".join(REPORT))


if __name__ == "__main__":
    main()
