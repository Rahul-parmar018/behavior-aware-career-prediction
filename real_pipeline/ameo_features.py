"""
Behavioral feature construction on the real AMEO 2015 dataset.

Source: Aspiring Minds Employability Outcomes 2015 (AMEO), 3,998 Indian
engineering graduates. Aggarwal et al., ACM CODS 2016,
doi:10.1145/2888451.2892037.

Defines, per student:
  ASR  Academic Shock Response  - trajectory-aware response to the two academic
       transitions (secondary -> higher secondary -> university), Eq. (1)-(3)
  TSR  Time Shock Response      - pace retention: years lost across the same
       transitions, Eq. (4)-(5)
  SSR  Skill Shock Response     - independently tested skill beyond what the
       academic record predicts, Eq. (6)-(7), plus domain-specific SSR
       components used by the validation layer
  CSS  Career Survivability Score - weighted summary, Eq. (8); NOT a classifier
       input (it is a deterministic function of ASR/SSR/TSR)

Run:  python real_pipeline/ameo_features.py
Out:  data/real/ameo_features.csv
"""
import os
import re
import sys

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

sys.path.insert(0, os.path.dirname(__file__))
from career_mapping import map_designation  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW = os.path.join(ROOT, "data", "real", "ameo_2015.xlsx")
OUT = os.path.join(ROOT, "data", "real", "ameo_features.csv")

SNAPSHOT = pd.Timestamp("2015-12-01")   # latest DOJ/DOL in the release
EXPECTED_AGE_12 = 18                    # age at higher-secondary completion
EXPECTED_DEGREE_YEARS = {"B.Tech/B.E.": 4, "MCA": 6, "M.Tech./M.E.": 6,
                         "M.Sc. (Tech.)": 5}

# ASR hyper-parameters (sensitivity-tested in experiments)
SHOCK_TAU = 0.5      # a drop of more than 0.5 SD in relative standing = shock
RECOVERY_LAMBDA = 1.0

SKILL_MODULES = ["ComputerProgramming", "ComputerScience",
                 "ElectronicsAndSemicon", "MechanicalEngg", "ElectricalEngg",
                 "TelecomEngg", "CivilEngg"]
APTITUDE = ["English", "Logical", "Quant"]


def _zscore_within(values, groups, min_group=30):
    """z-score within groups (e.g. exam board); small groups use global stats."""
    counts = groups.map(groups.value_counts())
    g = groups.where(counts >= min_group, "__other__")
    mu = values.groupby(g).transform("mean")
    sd = values.groupby(g).transform("std").replace(0, np.nan)
    return ((values - mu) / sd).fillna(0.0)


def _pct(x):
    """Percentile rank in [0, 1]."""
    return x.rank(pct=True)


def clean(df):
    df = df.copy()
    df["Designation"] = df["Designation"].astype(str).str.lower().str.strip()
    # 12 rows report GPA on a 10-point scale
    df.loc[df["collegeGPA"] <= 10, "collegeGPA"] *= 10
    df["age12"] = df["12graduation"] - df["DOB"].dt.year
    df["degree_years"] = df["GraduationYear"] - df["12graduation"]
    ok = (df["GraduationYear"] > 2000) & df["age12"].between(14, 25) \
        & df["degree_years"].between(3, 12) & df["Degree"].isin(EXPECTED_DEGREE_YEARS)
    return df[ok].reset_index(drop=True)


def build_features(df, tau=SHOCK_TAU, lam=RECOVERY_LAMBDA):
    f = pd.DataFrame(index=df.index)

    # --- Relative standing at each academic stage (board / tier normalised) ---
    z10 = _zscore_within(df["10percentage"], df["10board"].astype(str).str.lower())
    z12 = _zscore_within(df["12percentage"], df["12board"].astype(str).str.lower())
    zg = _zscore_within(df["collegeGPA"],
                        df["CollegeTier"].astype(str) + "|" + df["Degree"])
    f["z10"], f["z12"], f["zG"] = z10, z12, zg

    # --- ASR: Eq. (1) transition deltas, (2) shock indicator, (3) response ---
    d1 = z12 - z10                     # response to transition shock 1
    d2 = zg - z12                      # response to transition shock 2
    shock = (d1 < -tau).astype(float)  # a setback at the first transition
    f["delta1"], f["delta2"], f["academic_shock"] = d1, d2, shock
    f["ASR_raw"] = (d1 + d2) / 2 + lam * shock * d2
    f["ASR"] = _pct(f["ASR_raw"])

    # --- TSR: Eq. (4) delays at each transition, (5) pace retention ---
    d_school = (df["age12"] - EXPECTED_AGE_12).clip(lower=0)
    d_degree = (df["degree_years"] - df["Degree"].map(EXPECTED_DEGREE_YEARS)).clip(lower=0)
    f["delay_school"], f["delay_degree"] = d_school, d_degree
    # a delay that follows an academic shock is weighted as propagated shock
    f["TSR_raw"] = 1.0 / (1.0 + d_school + d_degree * (1.0 + 0.5 * shock))
    f["TSR"] = f["TSR_raw"]

    # --- SSR: Eq. (6) skill level, (7) residual beyond the academic record ---
    skill_z = pd.DataFrame(index=df.index)
    for m in SKILL_MODULES + ["Domain"]:
        taken = df[m] != -1
        col = pd.Series(np.nan, index=df.index)
        col[taken] = (df.loc[taken, m] - df.loc[taken, m].mean()) / df.loc[taken, m].std()
        skill_z[m] = col
    f["skill_level"] = skill_z.mean(axis=1, skipna=True).fillna(0.0)
    X_acad = pd.concat([z10, z12, zg, df["CollegeTier"],
                        pd.get_dummies(df["Degree"], dtype=float)], axis=1).values
    reg = LinearRegression().fit(X_acad, f["skill_level"])
    f["SSR_raw"] = f["skill_level"] - reg.predict(X_acad)
    f["SSR"] = _pct(f["SSR_raw"])

    # domain-specific SSR components (percentiles; untaken module -> 0)
    for m in SKILL_MODULES:
        taken = df[m] != -1
        col = pd.Series(0.0, index=df.index)
        col[taken] = df.loc[taken, m].rank(pct=True)
        f[f"ssr_{m}"] = col
    for a in APTITUDE:
        f[f"ssr_{a}"] = df[a].rank(pct=True)

    # --- CSS: Eq. (8) equal-weight summary (weights validated later) ---
    f["CSS"] = (f["ASR"] + f["SSR"] + f["TSR"]) / 3
    return f


def outcomes(df):
    o = pd.DataFrame(index=df.index)
    dol = pd.to_datetime(df["DOL"].where(df["DOL"].astype(str) != "present"),
                         errors="coerce")
    left = dol.notna()
    end = dol.fillna(SNAPSHOT)
    o["tenure_months"] = (end - df["DOJ"]).dt.days / 30.44
    o["left_job"] = left.astype(int)
    # early attrition: left within 12 months; defined only for students
    # observable for at least 12 months (joined on/before 2014-12-01)
    observable = df["DOJ"] <= SNAPSHOT - pd.DateOffset(months=12)
    o["early_exit"] = np.where(observable,
                               (left & (o["tenure_months"] < 12)).astype(int), np.nan)
    o["log_salary"] = np.log(df["Salary"].clip(lower=35000))
    return o


def main():
    raw = pd.read_excel(RAW)
    df = clean(raw)
    df["career_domain"] = df["Designation"].map(map_designation)
    feats = build_features(df)
    outs = outcomes(df)
    keep = ["ID", "Designation", "career_domain", "Gender", "Degree",
            "Specialization", "CollegeTier", "10percentage", "12percentage",
            "collegeGPA", "Salary", "conscientiousness", "agreeableness",
            "extraversion", "nueroticism", "openess_to_experience"]
    full = pd.concat([df[keep], feats, outs], axis=1)
    full.to_csv(OUT, index=False)

    print(f"Raw rows: {len(raw)}  after cleaning: {len(df)}")
    print(f"Mapped to a career domain: {full['career_domain'].notna().sum()}  "
          f"(ambiguous / other titles excluded: {full['career_domain'].isna().sum()})")
    print(full["career_domain"].value_counts().to_string())
    print(f"Academic shock rate: {feats['academic_shock'].mean():.1%}")
    print(f"Any delay: {((feats['delay_school'] + feats['delay_degree']) > 0).mean():.1%}")
    print(feats[["ASR", "SSR", "TSR", "CSS"]].corr().round(3).to_string())
    print(f"Saved {OUT}")


if __name__ == "__main__":
    main()
