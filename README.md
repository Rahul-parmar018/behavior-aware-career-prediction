# Behavior-Aware Career Prediction Using Shock Response Modeling

Code, label mappings and results for the paper *"Behavior-Aware Career Prediction Using Shock Response
Modeling and Domain-Based Validation"* (R. Parmar and A. Acharya, L J University, Ahmedabad).

The paper defines three trajectory-based behavioral features and evaluates them on two real, public datasets:

| Feature | What it measures |
|---|---|
| **ASR**: Academic Shock Response | Change in relative standing across 10th → 12th → university, with an explicit recovery term after a setback |
| **TSR**: Time Shock Response | Time lost at each academic transition; delays that follow a setback weigh more |
| **SSR**: Skill Shock Response | Independently tested skill beyond what the academic record predicts |
| **CSS**: Career Survivability Score | Summary of the three, used for profiling only (not a classifier input) |

## Main results

| Task | Baseline (academic only) | With ASR/SSR/TSR |
|---|---|---|
| Career domain, AMEO 2015 (6 classes, n = 3,613), macro-F1 | 0.249 | 0.258 (McNemar p < 0.001; corrected CV t-test p = 0.17) |
| Dropout vs graduate, UCI (n = 3,630), ROC-AUC | 0.910 | 0.952 (corrected CV t-test p < 0.001) |

On UCI, students who recovered after an entry shock dropped out at 32.4%, against 70.3% for those who did not.
Full numbers are in [`results/`](results/) and figures in [`figures/`](figures/).

## Reproduce

```bash
pip install -r requirements.txt
# 1. download the two datasets into data/real/ (see data/README.md)
python real_pipeline/ameo_features.py      # builds data/real/ameo_features.csv
python real_pipeline/experiments_ameo.py   # classification, validation layer, clustering, CSS, sensitivity (~7 min)
python real_pipeline/experiments_uci.py    # generalization to dropout prediction (~1 min)
python real_pipeline/imputation_ameo.py    # mean / KNN / MICE / GAIN comparison (~25 min, needs torch)
python real_pipeline/make_pipeline_figure.py
python real_pipeline/make_paper_figures.py
```

Outputs are written to `outputs/real/`. The files in `results/` and `figures/` are the outputs reported in the
paper (random seed 42).

## Files

| Path | Contents |
|---|---|
| `real_pipeline/ameo_features.py` | Cleaning and ASR / TSR / SSR / CSS construction for AMEO 2015 |
| `real_pipeline/career_mapping.py` | Ordered keyword rules mapping 419 job titles to 6 career domains |
| `real_pipeline/experiments_ameo.py` | Baselines, bootstrap CIs, McNemar and corrected t-tests, validation layer, K-Means validation, CSS weights, Cohen's d, sensitivity |
| `real_pipeline/experiments_uci.py` | Same features on the UCI dropout dataset |
| `real_pipeline/imputation_ameo.py` | Imputation comparison, including a GAIN implementation |

## Data

The datasets are not redistributed here; see [`data/README.md`](data/README.md) for download links.

- AMEO 2015: V. Aggarwal, S. Srikant and H. Nisar, *Proc. 3rd IKDD CODS*, 2016, doi:10.1145/2888451.2892037
- UCI #697: V. Realinho et al., *Data* 7(11):146, 2022, doi:10.3390/data7110146 (CC BY 4.0)

## License

Code: MIT (see `LICENSE`). The datasets remain under their original terms.
