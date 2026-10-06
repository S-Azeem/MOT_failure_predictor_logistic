"""
All-vehicle model, step 2: fit, calibrate and export.

  1. Fit logistic regression on TRAIN (tests before VALIDATION_START)
  2. Check every coefficient's sign makes sense
  3. Score VALIDATION (VALIDATION_START to TEST_START) against an age-only baseline
  4. Calibrate on validation with Platt scaling
  5. Set risk-band cut-offs: 40th and 80th percentiles of probability, per age
  6. Export everything the Worker needs to outputs/model.json

The test period is not touched here - that's validate.py, run once.

Run from the repository root:
    python -m pipeline.failure_model.train_final
"""
import json
from datetime import date

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, roc_auc_score

from config import OUTPUTS, SAMPLE_PCT, TEST_START, TRAINING_ALL_CSV, VALIDATION_START
from pipeline.failure_model.scoring import (NUMERIC, ZERO_IF_NO_HISTORY, age_key,
                                            design_matrix, raw_logit)
from pipeline.vehicle_rules import rules_for_export

N_MAKES = 20
TARGET = "target_failed"
# Expected direction of each numeric feature: more of it means more risk
EXPECTED_POSITIVE = ["age_years", "n_prior_fails", "fails_last_3", "prev_odometer",
                     "prev_fail_items", "prev_advisories"]


def brier_skill(y, p, base_rate):
    return 1 - brier_score_loss(y, p) / brier_score_loss(y, np.full(len(y), base_rate))


def main() -> None:
    df = pd.read_csv(TRAINING_ALL_CSV, dtype={"make": str})
    train = df[df["test_date"] < VALIDATION_START]
    val = df[(df["test_date"] >= VALIDATION_START) & (df["test_date"] < TEST_START)]
    print(f"Train: {len(train):,} cycles, fail rate {100 * train[TARGET].mean():.1f}%")
    print(f"Validation: {len(val):,} cycles, fail rate {100 * val[TARGET].mean():.1f}%\n")

    # ---- Preprocessing learned from TRAIN only ----------------------------------
    has_history = train["n_prior_fails"].notna()
    fill_values, means, sds = {}, {}, {}
    for c in NUMERIC:
        if c in ZERO_IF_NO_HISTORY:
            fill_values[c] = 0.0
            filled = train[c].fillna(0.0)
        else:
            # median over cars that have history, so first-time cars don't drag it
            fill_values[c] = float(train.loc[has_history, c].median()
                                   if c != "age_years" else train[c].median())
            filled = train[c].fillna(fill_values[c])
        means[c], sds[c] = float(filled.mean()), float(filled.std())
    makes = train["make"].value_counts().head(N_MAKES).index.tolist()

    columns = NUMERIC + ["no_history", "odometer_missing"] + [f"make_{m}" for m in makes]
    model = {
        "preprocessing": {"fill_values": fill_values, "means": means, "sds": sds,
                          "makes": makes},
        "columns": columns,
    }

    # ---- 1. Fit --------------------------------------------------------------------
    X_train = design_matrix(train, model)
    lr = LogisticRegression(C=1.0, max_iter=5000).fit(X_train, train[TARGET])
    model["coefficients"] = lr.coef_[0].tolist()
    model["intercept"] = float(lr.intercept_[0])

    # ---- 2. Sign check ---------------------------------------------------------------
    coefs = pd.Series(model["coefficients"], index=columns)
    print("Coefficients (numeric features per standard deviation; flags and makes per 0/1):")
    print("  feature                    coef   odds ratio")
    for c in columns:
        flag = ""
        if c in EXPECTED_POSITIVE and coefs[c] <= 0:
            flag = "  <- UNEXPECTED SIGN"
        print(f"  {c:<24} {coefs[c]:+7.3f}   x{np.exp(coefs[c]):.2f}{flag}")
    print(f"  (make reference = all makes outside the top {N_MAKES})\n")

    # ---- 3. Validation scores vs an age-only baseline ----------------------------------
    base_rate = float(train[TARGET].mean())
    z_val = raw_logit(val, model)
    p_raw = 1 / (1 + np.exp(-z_val))
    age_rates = train.groupby(train["age_years"].apply(age_key))[TARGET].mean()
    p_base = val["age_years"].apply(age_key).map(age_rates).fillna(base_rate).to_numpy()
    y_val = val[TARGET].to_numpy()

    # ---- 4. Platt scaling on validation -------------------------------------------------
    platt = LogisticRegression(C=1e6, max_iter=1000).fit(z_val.reshape(-1, 1), y_val)
    a, b = float(platt.coef_[0][0]), float(platt.intercept_[0])
    model["calibration"] = {"method": "platt", "a": a, "b": b}
    p_cal = 1 / (1 + np.exp(-(a * z_val + b)))

    print("Validation performance:")
    print(f"  {'model':<34} {'AUC':>6} {'Brier skill':>12}")
    for name, p in [("Baseline (age only)", p_base), ("Logistic, raw", p_raw),
                    ("Logistic, calibrated (in-sample)", p_cal)]:
        print(f"  {name:<34} {roc_auc_score(y_val, p):6.3f} {brier_skill(y_val, p, base_rate):12.3f}")
    print(f"\n  Calibration slope {a:.3f} (gate 0.9 to 1.1), intercept {b:+.3f}")
    print("  (the calibrated row is scored on the data it was fitted to; the honest check is validate.py)\n")

    # ---- 5. Risk-band cut-offs per age, from validation (the most recent year) ----------
    ages = val["age_years"].apply(age_key)
    overall = [float(np.quantile(p_cal, 0.4)), float(np.quantile(p_cal, 0.8))]
    cutoffs = {}
    for k in sorted(set(ages) | {str(a) for a in range(3, 25)} | {"25+"},
                    key=lambda s: 99 if s == "25+" else int(s)):
        group = p_cal[(ages == k).to_numpy()]
        cutoffs[k] = ([float(np.quantile(group, 0.4)), float(np.quantile(group, 0.8))]
                      if len(group) >= 200 else overall)
    model["bands"] = {"method": "40th and 80th percentile of calibrated probability among "
                                "cars of the same age, validation year",
                      "cutoffs_by_age": cutoffs}

    bands = np.where(p_cal < [cutoffs[k][0] for k in ages], "Lower risk",
                     np.where(p_cal < [cutoffs[k][1] for k in ages], "Typical", "Higher risk"))
    print("Fail rate by risk band (validation):")
    for b_name in ["Lower risk", "Typical", "Higher risk"]:
        m = bands == b_name
        print(f"  {b_name:<12} {100 * y_val[m].mean():5.1f}%  ({m.sum():,} cars)")

    # ---- 6. Export ----------------------------------------------------------------------
    model.update({
        "name": "Roadworthy MOT failure model",
        "version": "1.0",
        "built": date.today().isoformat(),
        "scope": "Cars and light vans, Great Britain; motorcycles excluded",
        "vehicle_rules": rules_for_export(),
        "training": {"sample_pct": SAMPLE_PCT, "train_cycles": len(train),
                     "validation_cycles": len(val), "train_fail_rate": base_rate,
                     "validation_auc_raw": float(roc_auc_score(y_val, p_raw))},
    })
    OUTPUTS.mkdir(parents=True, exist_ok=True)
    out = OUTPUTS / "model.json"
    out.write_text(json.dumps(model, indent=2))
    print(f"\nSaved {out}")


if __name__ == "__main__":
    main()