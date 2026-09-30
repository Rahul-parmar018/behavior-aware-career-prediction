"""
All experiments requested by the ICICSCS-2026 reviewers, on real AMEO 2015 data.

  E1  Classification: baselines vs behavioral vs proposed, 3 models,
      held-out test set with bootstrap 95% CIs, McNemar test,
      repeated 5x5 CV with corrected resampled t-test, fit/predict time
  E2  Full test-set outcome: per-class report, confusion matrix, failure cases
  E3  CSS redundancy ablation and permutation feature importance
  E4  Domain-consistency validation layer: threshold calibrated on training
      folds only, then coverage / accepted accuracy on test
  E5  K-Means risk clusters: silhouette / Davies-Bouldin / Calinski-Harabasz
      for k=2..6, bootstrap stability (ARI), external validation vs job outcomes
  E6  CSS weighting: equal vs learned vs weight grid, validated on job outcomes;
      Cohen's d with means, SDs, pooled-SD formula and bootstrap CI
  E7  Sensitivity: input noise, shock threshold tau, recovery weight lambda

Run:  python real_pipeline/experiments_ameo.py
Out:  outputs/real/*.png, outputs/real/ameo_results.json, outputs/real/ameo_report.txt
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
from scipy import stats
from sklearn.cluster import KMeans
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, adjusted_rand_score, balanced_accuracy_score,
                             calinski_harabasz_score, classification_report,
                             cohen_kappa_score, confusion_matrix, davies_bouldin_score,
                             f1_score, roc_auc_score, silhouette_score)
from sklearn.model_selection import (RepeatedStratifiedKFold, StratifiedKFold,
                                     cross_val_predict, cross_val_score, train_test_split)
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(__file__))
import ameo_features as af  # noqa: E402
from career_mapping import DOMAINS, map_designation  # noqa: E402

ROOT = af.ROOT
OUT = os.path.join(ROOT, "outputs", "real")
os.makedirs(OUT, exist_ok=True)
SEED = 42
N_BOOT = 1000
rng = np.random.default_rng(SEED)
REPORT = []
RESULTS = {}


def log(s=""):
    print(s)
    REPORT.append(s)


# ─────────────────────────────── data ────────────────────────────────────────
def spec_group(s):
    s = str(s).lower()
    if "computer application" in s:
        return "MCA"
    if any(k in s for k in ["computer", "information", "software"]):
        return "CS_IT"
    if "electrical" in s and "electronics" not in s:
        return "EE"
    if any(k in s for k in ["electronics", "telecom", "instrumentation", "communication"]):
        return "ECE"
    if any(k in s for k in ["mechanical", "production", "industrial", "automobile", "aeronautical"]):
        return "MECH"
    if "civil" in s:
        return "CIVIL"
    return "OTHER"


def load(tau=af.SHOCK_TAU, lam=af.RECOVERY_LAMBDA, noise=0.0, seed=SEED):
    raw = pd.read_excel(af.RAW)
    df = af.clean(raw)
    if noise > 0:  # sensitivity: perturb marks and skill scores
        r = np.random.default_rng(seed)
        for c in ["10percentage", "12percentage", "collegeGPA"]:
            df[c] = (df[c] + r.normal(0, noise, len(df))).clip(0, 100)
        for c in af.SKILL_MODULES + af.APTITUDE:
            taken = df[c] != -1
            sd = df.loc[taken, c].std()
            df.loc[taken, c] = df.loc[taken, c] + r.normal(0, noise / 10 * sd, taken.sum())
    df["career_domain"] = df["Designation"].map(map_designation)
    feats = af.build_features(df, tau=tau, lam=lam)
    outs = af.outcomes(df)
    df = pd.concat([df, feats, outs], axis=1)
    df["spec_group"] = df["Specialization"].map(spec_group)
    return df


BEH_CORE = ["ASR", "SSR", "TSR"]
SSR_COMPONENTS = [f"ssr_{m}" for m in af.SKILL_MODULES + af.APTITUDE]
PERSONALITY = ["conscientiousness", "agreeableness", "extraversion", "nueroticism",
               "openess_to_experience"]


def academic_matrix(df):
    return pd.concat([df[["10percentage", "12percentage", "collegeGPA", "CollegeTier"]],
                      pd.get_dummies(df["Degree"], prefix="deg", dtype=float),
                      pd.get_dummies(df["spec_group"], prefix="spec", dtype=float)], axis=1)


def feature_sets(df):
    A = academic_matrix(df)
    B = df[BEH_CORE + SSR_COMPONENTS]
    return {
        "Academic (CGPA + marks + specialization)": A,
        "Behavioral core (ASR, SSR, TSR)": df[BEH_CORE],
        "Behavioral (ASR, SSR, TSR + domain SSR)": B,
        "Proposed (Academic + Behavioral)": pd.concat([A, B], axis=1),
        "Proposed + CSS (redundancy check)": pd.concat([A, B, df[["CSS"]]], axis=1),
        "Proposed + personality (extended)": pd.concat([A, B, df[PERSONALITY]], axis=1),
    }


def models():
    return {
        "RandomForest": RandomForestClassifier(n_estimators=400, min_samples_leaf=3,
                                               class_weight="balanced_subsample",
                                               n_jobs=-1, random_state=SEED),
        "LogisticRegression": make_pipeline(StandardScaler(), LogisticRegression(
            max_iter=3000, class_weight="balanced", C=0.5)),
        "HistGradientBoosting": HistGradientBoostingClassifier(
            max_iter=300, learning_rate=0.05, class_weight="balanced", random_state=SEED),
    }


def metrics(y, p):
    return {"accuracy": accuracy_score(y, p),
            "balanced_accuracy": balanced_accuracy_score(y, p),
            "macro_f1": f1_score(y, p, average="macro"),
            "kappa": cohen_kappa_score(y, p)}


def bootstrap_ci(y, p, n=N_BOOT):
    y, p = np.asarray(y), np.asarray(p)
    samples = {k: [] for k in ["accuracy", "balanced_accuracy", "macro_f1", "kappa"]}
    for _ in range(n):
        i = rng.integers(0, len(y), len(y))
        for k, v in metrics(y[i], p[i]).items():
            samples[k].append(v)
    return {k: (float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5)))
            for k, v in samples.items()}


def mcnemar(y, pa, pb):
    a_ok, b_ok = pa == y, pb == y
    n01 = int(np.sum(a_ok & ~b_ok))
    n10 = int(np.sum(~a_ok & b_ok))
    if n01 + n10 == 0:
        return 0.0, 1.0, n01, n10
    chi2 = (abs(n01 - n10) - 1) ** 2 / (n01 + n10)
    return float(chi2), float(1 - stats.chi2.cdf(chi2, 1)), n01, n10


def corrected_ttest(a, b, n_train, n_test):
    """Nadeau & Bengio corrected resampled t-test for repeated CV scores."""
    d = np.asarray(b) - np.asarray(a)
    k = len(d)
    var = np.var(d, ddof=1)
    t = d.mean() / np.sqrt((1 / k + n_test / n_train) * var) if var > 0 else 0.0
    p = 2 * (1 - stats.t.cdf(abs(t), k - 1))
    return float(t), float(p)


# ───────────────────────── E1 / E2 / E3 classification ───────────────────────
def run_classification(df):
    log("=" * 78)
    log("E1  CAREER-DOMAIN CLASSIFICATION  (real AMEO 2015 data)")
    log("=" * 78)
    d = df[df["career_domain"].notna()].reset_index(drop=True)
    y = d["career_domain"].values
    log(f"Students with a mappable career domain: {len(d)}  |  classes: {len(DOMAINS)}")
    for c, n in d["career_domain"].value_counts().items():
        log(f"   {c:<24s} {n:5d}  ({n / len(d):.1%})")
    idx_tr, idx_te = train_test_split(np.arange(len(d)), test_size=0.2,
                                      stratify=y, random_state=SEED)
    log(f"Split: stratified 80/20 -> train {len(idx_tr)}, test {len(idx_te)} (seed {SEED})")
    y_tr, y_te = y[idx_tr], y[idx_te]

    # trivial baselines
    maj = pd.Series(y_tr).mode()[0]
    p_maj = np.array([maj] * len(y_te))
    probs = pd.Series(y_tr).value_counts(normalize=True)
    p_strat = rng.choice(probs.index, size=len(y_te), p=probs.values)
    base = {"Majority class": metrics(y_te, p_maj), "Stratified random": metrics(y_te, p_strat)}
    log("\nTrivial baselines (test set):")
    for k, m in base.items():
        log(f"   {k:<22s} acc={m['accuracy']:.3f}  bal_acc={m['balanced_accuracy']:.3f}  "
            f"macroF1={m['macro_f1']:.3f}  kappa={m['kappa']:.3f}")

    fsets = feature_sets(d)
    table, preds, fitted = [], {}, {}
    for fs_name, X in fsets.items():
        X = X.values.astype(float)
        for m_name, model in models().items():
            t0 = time.perf_counter()
            model.fit(X[idx_tr], y_tr)
            t_fit = time.perf_counter() - t0
            t0 = time.perf_counter()
            p = model.predict(X[idx_te])
            t_pred = (time.perf_counter() - t0) / len(idx_te) * 1000
            m = metrics(y_te, p)
            ci = bootstrap_ci(y_te, p)
            preds[(fs_name, m_name)] = p
            fitted[(fs_name, m_name)] = model
            table.append({"features": fs_name, "model": m_name, **m,
                          "ci": ci, "fit_s": t_fit, "pred_ms_per_student": t_pred})

    log("\nHeld-out test set, 95% bootstrap CI in brackets (1000 resamples):")
    log(f"{'Feature set':<42s}{'Model':<22s}{'Acc':>16s}{'Bal.Acc':>16s}{'Macro-F1':>16s}"
        f"{'Kappa':>16s}{'Fit s':>7s}")
    for r in table:
        def f(k):
            return f"{r[k]:.3f}[{r['ci'][k][0]:.2f},{r['ci'][k][1]:.2f}]"
        log(f"{r['features']:<42s}{r['model']:<22s}{f('accuracy'):>16s}"
            f"{f('balanced_accuracy'):>16s}{f('macro_f1'):>16s}{f('kappa'):>16s}{r['fit_s']:7.2f}")

    # McNemar: academic vs proposed, per model
    log("\nMcNemar test, Academic vs Proposed (same model, same test students):")
    mc = {}
    for m_name in models():
        pa = preds[("Academic (CGPA + marks + specialization)", m_name)]
        pb = preds[("Proposed (Academic + Behavioral)", m_name)]
        chi2, p, n01, n10 = mcnemar(y_te, pa, pb)
        mc[m_name] = {"chi2": chi2, "p": p, "academic_only_correct": n01,
                      "proposed_only_correct": n10}
        log(f"   {m_name:<22s} chi2={chi2:.3f}  p={p:.4g}  "
            f"(only academic correct={n01}, only proposed correct={n10})")

    # repeated CV on macro-F1 with corrected t-test (RandomForest, primary model)
    log("\nRepeated stratified 5x5 CV, RandomForest, macro-F1 (mean +/- SD):")
    rcv = RepeatedStratifiedKFold(n_splits=5, n_repeats=5, random_state=SEED)
    cv_scores = {}
    for fs_name in ["Academic (CGPA + marks + specialization)",
                    "Behavioral (ASR, SSR, TSR + domain SSR)",
                    "Proposed (Academic + Behavioral)",
                    "Proposed + CSS (redundancy check)"]:
        s = cross_val_score(models()["RandomForest"], fsets[fs_name].values.astype(float), y,
                            cv=rcv, scoring="f1_macro", n_jobs=-1)
        cv_scores[fs_name] = s
        log(f"   {fs_name:<42s} {s.mean():.4f} +/- {s.std(ddof=1):.4f}")
    n_te = len(d) // 5
    t, p = corrected_ttest(cv_scores["Academic (CGPA + marks + specialization)"],
                           cv_scores["Proposed (Academic + Behavioral)"], len(d) - n_te, n_te)
    log(f"   Corrected resampled t-test, Proposed vs Academic: t={t:.3f}, p={p:.4g}")
    t2, p2 = corrected_ttest(cv_scores["Proposed (Academic + Behavioral)"],
                             cv_scores["Proposed + CSS (redundancy check)"], len(d) - n_te, n_te)
    log(f"   Corrected resampled t-test, Proposed+CSS vs Proposed: t={t2:.3f}, p={p2:.4g}")

    RESULTS["classification"] = {
        "n": len(d), "class_counts": d["career_domain"].value_counts().to_dict(),
        "trivial_baselines": base, "table": table, "mcnemar": mc,
        "cv_macro_f1": {k: {"mean": float(v.mean()), "sd": float(v.std(ddof=1))}
                        for k, v in cv_scores.items()},
        "cv_ttest_proposed_vs_academic": {"t": t, "p": p},
        "cv_ttest_css_vs_proposed": {"t": t2, "p": p2},
    }

    # E2 full test-set outcome for the primary configuration
    key = ("Proposed (Academic + Behavioral)", "RandomForest")
    p = preds[key]
    log("\n" + "=" * 78)
    log("E2  FULL TEST-SET OUTCOME  (Proposed, RandomForest)")
    log("=" * 78)
    log(classification_report(y_te, p, labels=DOMAINS, digits=3, zero_division=0))
    cm = confusion_matrix(y_te, p, labels=DOMAINS)
    fig, ax = plt.subplots(figsize=(7.5, 6))
    im = ax.imshow(cm / cm.sum(axis=1, keepdims=True), cmap="Blues", vmin=0, vmax=1)
    for i in range(len(DOMAINS)):
        for j in range(len(DOMAINS)):
            ax.text(j, i, cm[i, j], ha="center", va="center", fontsize=9,
                    color="white" if cm[i, j] / cm[i].sum() > 0.5 else "black")
    short = [s.replace("_", "\n") for s in DOMAINS]
    ax.set_xticks(range(len(DOMAINS)), short, fontsize=8)
    ax.set_yticks(range(len(DOMAINS)), short, fontsize=8)
    ax.set_xlabel("Predicted domain")
    ax.set_ylabel("Actual domain")
    fig.colorbar(im, ax=ax, label="Row-normalised rate")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "fig_confusion_matrix.png"), dpi=300)
    plt.close(fig)
    wrong = pd.Series([f"{a} -> {b}" for a, b in zip(y_te, p) if a != b]).value_counts()
    log("Most frequent failure cases (actual -> predicted):")
    for k, v in wrong.head(6).items():
        log(f"   {k:<50s} {v}")
    RESULTS["test_confusion_matrix"] = {"labels": DOMAINS, "matrix": cm.tolist()}
    RESULTS["failure_cases"] = wrong.head(10).to_dict()

    # E3 permutation importance
    log("\n" + "=" * 78)
    log("E3  PERMUTATION FEATURE IMPORTANCE (test set, macro-F1 drop, 20 repeats)")
    log("=" * 78)
    Xp = fsets["Proposed (Academic + Behavioral)"]
    pi = permutation_importance(fitted[key], Xp.values[idx_te].astype(float), y_te,
                                scoring="f1_macro", n_repeats=20, random_state=SEED, n_jobs=-1)
    imp = pd.DataFrame({"feature": Xp.columns, "mean": pi.importances_mean,
                        "sd": pi.importances_std}).sort_values("mean", ascending=False)
    for _, r in imp.head(15).iterrows():
        log(f"   {r['feature']:<28s} {r['mean']:+.4f} +/- {r['sd']:.4f}")
    RESULTS["permutation_importance"] = imp.to_dict("records")
    top = imp.head(15).iloc[::-1]
    fig, ax = plt.subplots(figsize=(6.5, 5))
    colors = ["#c0392b" if f in BEH_CORE else ("#e67e22" if f.startswith("ssr_") else "#7f8c8d")
              for f in top["feature"]]
    ax.barh(top["feature"], top["mean"], xerr=top["sd"], color=colors)
    ax.set_xlabel("Drop in macro-F1 when permuted")
    ax.set_title("Permutation importance (red: ASR/SSR/TSR, orange: domain SSR)", fontsize=9)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "fig_feature_importance.png"), dpi=300)
    plt.close(fig)

    return d, idx_tr, idx_te, fsets


# ───────────────────────── E4 validation layer ───────────────────────────────
DOMAIN_SKILLS = {
    "Software_Dev": ["ssr_ComputerProgramming", "ssr_ComputerScience"],
    "QA_Testing": ["ssr_ComputerProgramming", "ssr_Logical"],
    "IT_Infra_Support": ["ssr_ComputerScience", "ssr_TelecomEngg", "ssr_ComputerProgramming"],
    "Data_Business_Analysis": ["ssr_Quant", "ssr_Logical"],
    "Core_Engineering": ["ssr_ElectronicsAndSemicon", "ssr_MechanicalEngg",
                         "ssr_ElectricalEngg", "ssr_TelecomEngg", "ssr_CivilEngg"],
    "Management_Sales_Ops": ["ssr_English"],
}


def domain_ssr(rows, predicted):
    return np.array([rows.iloc[i][DOMAIN_SKILLS[c]].max() for i, c in enumerate(predicted)])


def run_validation_layer(d, idx_tr, idx_te, fsets):
    log("\n" + "=" * 78)
    log("E4  DOMAIN-CONSISTENCY VALIDATION LAYER")
    log("=" * 78)
    X = fsets["Proposed (Academic + Behavioral)"].values.astype(float)
    y = d["career_domain"].values
    rf = models()["RandomForest"]
    # calibrate threshold on out-of-fold predictions of the TRAINING set only
    oof = cross_val_predict(rf, X[idx_tr], y[idx_tr],
                            cv=StratifiedKFold(5, shuffle=True, random_state=SEED), n_jobs=-1)
    s_tr = domain_ssr(d.iloc[idx_tr].reset_index(drop=True), oof)
    grid = np.round(np.arange(0.0, 0.95, 0.05), 2)
    rows = []
    for th in grid:
        acc_mask = s_tr >= th
        if acc_mask.sum() < 30:
            continue
        rows.append({"threshold": th, "coverage": acc_mask.mean(),
                     "accepted_macro_f1": f1_score(y[idx_tr][acc_mask], oof[acc_mask],
                                                   average="macro"),
                     "accepted_acc": accuracy_score(y[idx_tr][acc_mask], oof[acc_mask])})
    cal = pd.DataFrame(rows)
    # rule: highest accepted macro-F1 while keeping >= 70% coverage
    eligible = cal[cal["coverage"] >= 0.70]
    th_star = float(eligible.loc[eligible["accepted_macro_f1"].idxmax(), "threshold"])
    log(f"Threshold calibrated on training folds (max accepted macro-F1 s.t. coverage>=70%): "
        f"SSR_domain >= {th_star:.2f}")

    rf.fit(X[idx_tr], y[idx_tr])
    p_te = rf.predict(X[idx_te])
    s_te = domain_ssr(d.iloc[idx_te].reset_index(drop=True), p_te)
    y_te = y[idx_te]
    ok = s_te >= th_star
    res = {"threshold": th_star, "coverage": float(ok.mean()),
           "rejection_rate": float(1 - ok.mean()),
           "all": metrics(y_te, p_te),
           "accepted": metrics(y_te[ok], p_te[ok]),
           "rejected": metrics(y_te[~ok], p_te[~ok]) if (~ok).sum() else None}
    # comparison: confidence gating with RF probability at the same coverage
    proba = rf.predict_proba(X[idx_te]).max(axis=1)
    cut = np.quantile(proba, 1 - ok.mean())
    okp = proba >= cut
    res["prob_gate_same_coverage"] = metrics(y_te[okp], p_te[okp])
    log(f"Test set: coverage={res['coverage']:.1%}, rejected={res['rejection_rate']:.1%}")
    for k in ["all", "accepted", "rejected", "prob_gate_same_coverage"]:
        m = res[k]
        if m:
            log(f"   {k:<26s} acc={m['accuracy']:.3f}  bal_acc={m['balanced_accuracy']:.3f}  "
                f"macroF1={m['macro_f1']:.3f}")
    # test-set sweep for the figure (reported, not used for choosing)
    sweep = []
    for th in grid:
        mk = s_te >= th
        if mk.sum() < 20:
            continue
        sweep.append({"threshold": float(th), "coverage": float(mk.mean()),
                      "accepted_acc": accuracy_score(y_te[mk], p_te[mk]),
                      "accepted_macro_f1": f1_score(y_te[mk], p_te[mk], average="macro")})
    sw = pd.DataFrame(sweep)
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(sw["threshold"], sw["accepted_acc"], "o-", label="Accuracy of accepted")
    ax.plot(sw["threshold"], sw["accepted_macro_f1"], "s-", label="Macro-F1 of accepted")
    ax.plot(sw["threshold"], sw["coverage"], "^--", color="gray", label="Coverage")
    ax.axvline(th_star, color="red", ls=":", label=f"Calibrated threshold {th_star:.2f}")
    ax.set_xlabel("Domain-SSR threshold")
    ax.set_ylabel("Rate")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "fig_threshold_sweep.png"), dpi=300)
    plt.close(fig)
    res["test_sweep"] = sweep
    res["train_calibration"] = cal.to_dict("records")
    RESULTS["validation_layer"] = res


# ───────────────────────── E5 clustering ─────────────────────────────────────
def run_clustering(df):
    log("\n" + "=" * 78)
    log("E5  K-MEANS RISK CLUSTERS ON (ASR, SSR, TSR) - VALIDATION")
    log("=" * 78)
    Z = StandardScaler().fit_transform(df[BEH_CORE].values)
    sub = rng.choice(len(Z), size=min(3000, len(Z)), replace=False)
    scores = []
    for k in range(2, 7):
        km = KMeans(k, n_init=20, random_state=SEED).fit(Z)
        scores.append({"k": k, "silhouette": silhouette_score(Z[sub], km.labels_[sub]),
                       "davies_bouldin": davies_bouldin_score(Z, km.labels_),
                       "calinski_harabasz": calinski_harabasz_score(Z, km.labels_),
                       "inertia": km.inertia_})
    sc = pd.DataFrame(scores)
    log(sc.round(3).to_string(index=False))

    km = KMeans(3, n_init=20, random_state=SEED).fit(Z)
    lab = km.labels_
    aris = []
    for b in range(50):
        i = rng.integers(0, len(Z), len(Z))
        kb = KMeans(3, n_init=10, random_state=b).fit(Z[i])
        aris.append(adjusted_rand_score(lab, kb.predict(Z)))
    log(f"Bootstrap stability k=3 (50 resamples): ARI = {np.mean(aris):.3f} +/- {np.std(aris):.3f}")

    # Name clusters from their centroids, not by hand: the highest-CSS cluster is
    # "Resilient"; every other cluster is named after the feature that is most
    # below the population mean (in SD units). Outcomes are checked afterwards.
    cent = pd.DataFrame(km.cluster_centers_, columns=BEH_CORE)
    best = df.groupby(lab)["CSS"].mean().idxmax()
    labels_map = {"ASR": "Academic-shock", "SSR": "Skill-deficit", "TSR": "Time-delayed"}
    names = {best: "Resilient"}
    for c in cent.index:
        if c != best:
            names[c] = labels_map[cent.loc[c].idxmin()]
    risk = pd.Series(lab).map(names)
    df = df.assign(risk=risk.values)
    log("Cluster centroids (standardised) and data-derived names:")
    for c in cent.index:
        log(f"   {names[c]:<15s} " + "  ".join(f"{f}={cent.loc[c, f]:+.2f}" for f in BEH_CORE))
    prof = df.groupby("risk").agg(n=("CSS", "size"), ASR=("ASR", "mean"), SSR=("SSR", "mean"),
                                  TSR=("TSR", "mean"), CSS=("CSS", "mean"),
                                  early_exit_rate=("early_exit", "mean"),
                                  median_salary=("Salary", "median"),
                                  median_tenure_m=("tenure_months", "median"))
    prof = prof.sort_values("CSS", ascending=False)
    log("\nCluster profiles and EXTERNAL outcomes (not used for clustering):")
    log(prof.round(3).to_string())
    obs = df[df["early_exit"].notna()]
    ct = pd.crosstab(obs["risk"], obs["early_exit"])
    chi2, p_chi, _, _ = stats.chi2_contingency(ct)
    kw = stats.kruskal(*[g["log_salary"].values for _, g in df.groupby("risk")])
    log(f"Early exit vs cluster: chi2={chi2:.2f}, p={p_chi:.4g}")
    log(f"Salary vs cluster: Kruskal-Wallis H={kw.statistic:.2f}, p={kw.pvalue:.4g}")

    fig, axes = plt.subplots(1, 2, figsize=(9, 3.9))
    palette = ["#27ae60", "#f39c12", "#c0392b", "#8e44ad"]
    order = list(prof.index)
    jit = np.random.default_rng(SEED).normal(0, 0.015, len(df))
    for r, c in zip(order, palette):
        m = (df["risk"] == r).values
        axes[0].scatter(df.loc[m, "ASR"], df.loc[m, "SSR"], s=5, alpha=0.4, c=c,
                        label=f"{r} (n={m.sum()})")
        axes[1].scatter(df.loc[m, "SSR"], df.loc[m, "TSR"] + jit[m], s=5, alpha=0.4, c=c)
    axes[0].set_xlabel("Academic Shock Response (ASR)")
    axes[0].set_ylabel("Skill Shock Response (SSR)")
    axes[1].set_xlabel("Skill Shock Response (SSR)")
    axes[1].set_ylabel("Time Shock Response (TSR, jittered)")
    axes[0].legend(fontsize=8, markerscale=2.5)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "fig_risk_clusters.png"), dpi=300)
    plt.close(fig)

    RESULTS["clustering"] = {"k_scan": sc.to_dict("records"),
                             "ari_mean": float(np.mean(aris)), "ari_sd": float(np.std(aris)),
                             "profiles": prof.reset_index().to_dict("records"),
                             "early_exit_chi2": float(chi2), "early_exit_p": float(p_chi),
                             "salary_kw_H": float(kw.statistic), "salary_kw_p": float(kw.pvalue)}


# ───────────────────────── E6 CSS weights + Cohen's d ────────────────────────
def cohens_d(a, b):
    a, b = np.asarray(a), np.asarray(b)
    n1, n2 = len(a), len(b)
    sp = np.sqrt(((n1 - 1) * a.var(ddof=1) + (n2 - 1) * b.var(ddof=1)) / (n1 + n2 - 2))
    return (a.mean() - b.mean()) / sp


def boot_d(a, b, n=N_BOOT):
    ds = [cohens_d(rng.choice(a, len(a)), rng.choice(b, len(b))) for _ in range(n)]
    return float(np.percentile(ds, 2.5)), float(np.percentile(ds, 97.5))


def run_css(df):
    log("\n" + "=" * 78)
    log("E6  CSS WEIGHTING AND EFFECT SIZE (validated on real job outcomes)")
    log("=" * 78)
    obs = df[df["early_exit"].notna()].reset_index(drop=True)
    y_exit = obs["early_exit"].astype(int).values
    log(f"Early attrition sample (joined <= 2014-12, observable >= 12 months): n={len(obs)}, "
        f"exit rate={y_exit.mean():.1%}")

    def auc_of(w, frame, y):
        s = frame[BEH_CORE].values @ np.asarray(w)
        return roc_auc_score(y, -s)  # lower CSS -> higher risk

    eq = [1 / 3] * 3
    res = {"equal_weights": {"auc_early_exit": auc_of(eq, obs, y_exit),
                             "spearman_salary": float(stats.spearmanr(df["CSS"],
                                                                      df["log_salary"])[0])}}
    # learned weights, 5-fold CV (logistic regression on early exit)
    cv = StratifiedKFold(5, shuffle=True, random_state=SEED)
    lr = LogisticRegression(max_iter=1000)
    oof = cross_val_predict(lr, obs[BEH_CORE].values, y_exit, cv=cv, method="predict_proba")[:, 1]
    lr.fit(obs[BEH_CORE].values, y_exit)
    w = -lr.coef_[0]
    w_norm = w / np.abs(w).sum()
    res["learned_weights"] = {"weights_ASR_SSR_TSR": w_norm.tolist(),
                              "cv_auc_early_exit": roc_auc_score(y_exit, oof)}
    # weight grid on the simplex
    grid = []
    for a in np.arange(0, 1.01, 0.1):
        for s in np.arange(0, 1.01 - a, 0.1):
            t = 1 - a - s
            css = df[BEH_CORE].values @ np.array([a, s, t])
            grid.append({"w_ASR": round(a, 1), "w_SSR": round(s, 1), "w_TSR": round(t, 1),
                         "auc_early_exit": auc_of([a, s, t], obs, y_exit),
                         "spearman_salary": float(stats.spearmanr(css, df["log_salary"])[0])})
    g = pd.DataFrame(grid)
    res["weight_grid"] = grid
    log(f"Equal weights: AUC(early exit)={res['equal_weights']['auc_early_exit']:.3f}, "
        f"Spearman(CSS, salary)={res['equal_weights']['spearman_salary']:.3f}")
    log(f"Learned weights (ASR, SSR, TSR) = {np.round(w_norm, 3).tolist()}, "
        f"CV AUC(early exit)={res['learned_weights']['cv_auc_early_exit']:.3f}")
    log(f"Weight grid (66 combinations): AUC range {g['auc_early_exit'].min():.3f}-"
        f"{g['auc_early_exit'].max():.3f}; salary rho range {g['spearman_salary'].min():.3f}-"
        f"{g['spearman_salary'].max():.3f}")
    best = g.loc[g["spearman_salary"].idxmax()]
    log(f"Best salary weights: ASR={best.w_ASR}, SSR={best.w_SSR}, TSR={best.w_TSR}")

    # univariate feature validity
    log("\nEach behavioral feature vs outcomes:")
    uni = {}
    for f in BEH_CORE + ["CSS"]:
        rho, p = stats.spearmanr(df[f], df["log_salary"])
        auc = roc_auc_score(y_exit, -obs[f])
        uni[f] = {"spearman_salary": float(rho), "p_salary": float(p), "auc_early_exit": float(auc)}
        log(f"   {f:<4s} Spearman(salary) rho={rho:+.3f} (p={p:.3g})   AUC(early exit)={auc:.3f}")
    res["univariate"] = uni

    # Cohen's d - explicit groups, non-circular (outcome is not part of CSS)
    log("\nCohen's d, d = (M1 - M2) / s_p,  s_p = sqrt(((n1-1)s1^2 + (n2-1)s2^2)/(n1+n2-2))")
    q = df["CSS"].quantile([1 / 3, 2 / 3]).values
    hi = df.loc[df["CSS"] >= q[1], "log_salary"].values
    lo = df.loc[df["CSS"] <= q[0], "log_salary"].values
    d1 = cohens_d(hi, lo)
    ci1 = boot_d(hi, lo)
    log(f"   (a) log-salary, top-tertile CSS (n={len(hi)}, M={hi.mean():.3f}, SD={hi.std(ddof=1):.3f})"
        f" vs bottom-tertile CSS (n={len(lo)}, M={lo.mean():.3f}, SD={lo.std(ddof=1):.3f}):"
        f" d={d1:.3f}  95% CI [{ci1[0]:.3f}, {ci1[1]:.3f}]")
    stay = obs.loc[obs["early_exit"] == 0, "CSS"].values
    leave = obs.loc[obs["early_exit"] == 1, "CSS"].values
    d2 = cohens_d(stay, leave)
    ci2 = boot_d(stay, leave)
    log(f"   (b) CSS, retained >=12 months (n={len(stay)}, M={stay.mean():.3f}, SD={stay.std(ddof=1):.3f})"
        f" vs early exit (n={len(leave)}, M={leave.mean():.3f}, SD={leave.std(ddof=1):.3f}):"
        f" d={d2:.3f}  95% CI [{ci2[0]:.3f}, {ci2[1]:.3f}]")
    res["cohens_d"] = {
        "salary_top_vs_bottom_css": {"d": float(d1), "ci": ci1, "n1": len(hi), "n2": len(lo),
                                     "m1": float(hi.mean()), "sd1": float(hi.std(ddof=1)),
                                     "m2": float(lo.mean()), "sd2": float(lo.std(ddof=1))},
        "css_retained_vs_exit": {"d": float(d2), "ci": ci2, "n1": len(stay), "n2": len(leave),
                                 "m1": float(stay.mean()), "sd1": float(stay.std(ddof=1)),
                                 "m2": float(leave.mean()), "sd2": float(leave.std(ddof=1))}}
    # feature differences across career domains (effect size eta^2)
    d = df[df["career_domain"].notna()]
    log("\nOne-way ANOVA across career domains (eta^2 = effect size):")
    anova = {}
    for f in BEH_CORE:
        groups = [g[f].values for _, g in d.groupby("career_domain")]
        F, p = stats.f_oneway(*groups)
        grand = d[f].mean()
        ss_b = sum(len(g) * (g.mean() - grand) ** 2 for g in groups)
        ss_t = ((d[f] - grand) ** 2).sum()
        anova[f] = {"F": float(F), "p": float(p), "eta2": float(ss_b / ss_t)}
        log(f"   {f}: F={F:.2f}, p={p:.3g}, eta^2={ss_b / ss_t:.4f}")
    res["anova_by_domain"] = anova
    RESULTS["css"] = res


# ───────────────────────── E7 sensitivity ────────────────────────────────────
def run_sensitivity():
    log("\n" + "=" * 78)
    log("E7  SENSITIVITY ANALYSIS (RandomForest, Proposed features, macro-F1 on fixed split)")
    log("=" * 78)

    def score(df, seed=SEED):
        d = df[df["career_domain"].notna()].reset_index(drop=True)
        y = d["career_domain"].values
        X = feature_sets(d)["Proposed (Academic + Behavioral)"].values.astype(float)
        tr, te = train_test_split(np.arange(len(d)), test_size=0.2, stratify=y, random_state=SEED)
        m = RandomForestClassifier(n_estimators=200, min_samples_leaf=3,
                                   class_weight="balanced_subsample", n_jobs=-1,
                                   random_state=seed).fit(X[tr], y[tr])
        p = m.predict(X[te])
        return f1_score(y[te], p, average="macro"), accuracy_score(y[te], p)

    out = {"noise": [], "tau": [], "lambda": []}
    log("Input noise: Gaussian sigma (percentage points) added to 10th/12th/GPA marks;"
        " skill scores get sigma/10 x their SD. 10 repeats each.")
    for s in [0, 1, 2, 5, 10]:
        vals = [score(load(noise=s, seed=r), seed=r) for r in range(10 if s else 1)]
        f1s, accs = np.array([v[0] for v in vals]), np.array([v[1] for v in vals])
        out["noise"].append({"sigma": s, "macro_f1_mean": float(f1s.mean()),
                             "macro_f1_sd": float(f1s.std()), "acc_mean": float(accs.mean()),
                             "acc_sd": float(accs.std())})
        log(f"   sigma={s:>2d}: macro-F1={f1s.mean():.4f} +/- {f1s.std():.4f}   "
            f"acc={accs.mean():.4f} +/- {accs.std():.4f}")
    log("Shock threshold tau (SD units):")
    for t in [0.25, 0.5, 0.75, 1.0]:
        f1, acc = score(load(tau=t))
        out["tau"].append({"tau": t, "macro_f1": f1, "acc": acc})
        log(f"   tau={t:.2f}: macro-F1={f1:.4f}  acc={acc:.4f}")
    log("Recovery weight lambda:")
    for lam in [0.0, 0.5, 1.0, 2.0]:
        f1, acc = score(load(lam=lam))
        out["lambda"].append({"lambda": lam, "macro_f1": f1, "acc": acc})
        log(f"   lambda={lam:.1f}: macro-F1={f1:.4f}  acc={acc:.4f}")

    nz = pd.DataFrame(out["noise"])
    fig, ax = plt.subplots(figsize=(5.5, 3.8))
    ax.errorbar(nz["sigma"], nz["macro_f1_mean"], yerr=nz["macro_f1_sd"], fmt="o-",
                capsize=4, label="Macro-F1")
    ax.errorbar(nz["sigma"], nz["acc_mean"], yerr=nz["acc_sd"], fmt="s-", capsize=4,
                label="Accuracy")
    ax.set_xlabel("Noise sigma (percentage points)")
    ax.set_ylabel("Score (mean +/- SD, 10 runs)")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "fig_sensitivity_noise.png"), dpi=300)
    plt.close(fig)
    RESULTS["sensitivity"] = out


def feature_distribution_fig(df):
    fig, ax = plt.subplots(figsize=(6, 3.8))
    parts = ax.violinplot([df[f].values for f in BEH_CORE + ["CSS"]], showmedians=True)
    for pc, c in zip(parts["bodies"], ["#6c3483", "#b03a2e", "#d35400", "#1f618d"]):
        pc.set_facecolor(c)
        pc.set_alpha(0.7)
    ax.set_xticks([1, 2, 3, 4], ["ASR", "SSR", "TSR", "CSS"])
    ax.set_ylabel("Normalised score [0, 1]")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "fig_feature_distributions.png"), dpi=300)
    plt.close(fig)
    desc = df[BEH_CORE + ["CSS"]].describe().T
    log("\nFeature summary (all cleaned students):")
    log(desc.round(3).to_string())
    log("Feature correlations (Pearson):")
    log(df[BEH_CORE + ["CSS"]].corr().round(3).to_string())
    RESULTS["feature_summary"] = desc.to_dict()
    RESULTS["feature_corr"] = df[BEH_CORE + ["CSS"]].corr().to_dict()
    RESULTS["shock_rate"] = float(df["academic_shock"].mean())
    RESULTS["delay_rate"] = float(((df["delay_school"] + df["delay_degree"]) > 0).mean())


def main():
    t0 = time.time()
    df = load()
    log(f"AMEO 2015 cleaned students: {len(df)}")
    log(f"Academic shock rate: {df['academic_shock'].mean():.1%}; any time delay: "
        f"{((df['delay_school'] + df['delay_degree']) > 0).mean():.1%}")
    feature_distribution_fig(df)
    d, idx_tr, idx_te, fsets = run_classification(df)
    run_validation_layer(d, idx_tr, idx_te, fsets)
    run_clustering(df)
    run_css(df)
    run_sensitivity()
    log(f"\nTotal runtime: {time.time() - t0:.0f}s")
    with open(os.path.join(OUT, "ameo_results.json"), "w") as f:
        json.dump(RESULTS, f, indent=2, default=lambda o: o.item() if hasattr(o, "item") else str(o))
    with open(os.path.join(OUT, "ameo_report.txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(REPORT))


if __name__ == "__main__":
    main()
