"""
Score cars from the exported model JSON - the same file the Cloudflare Worker reads.

Every step here must be mirrored exactly in the Worker's JavaScript; the Day 4
parity test compares the two. Keep it plain: fill blanks, add two flags,
standardise, one-hot the make, sum, sigmoid, calibrate, pick a band.
"""
import numpy as np
import pandas as pd

# The seven numeric features, in model order
NUMERIC = ["age_years", "n_prior_fails", "fails_last_3", "prev_odometer",
           "prev_fail_items", "prev_advisories", "days_since_prev"]
# History counts: a car with no earlier test has none of these, so 0 is the true value
ZERO_IF_NO_HISTORY = ["n_prior_fails", "fails_last_3", "prev_fail_items", "prev_advisories"]


def design_matrix(df: pd.DataFrame, model: dict) -> np.ndarray:
    """Turn raw feature columns into the model's input matrix, in model["columns"] order."""
    prep = model["preprocessing"]
    X = pd.DataFrame(index=df.index)

    no_history = df["n_prior_fails"].isna()
    for c in NUMERIC:
        v = df[c].astype(float)
        if c in ZERO_IF_NO_HISTORY:
            v = v.fillna(0.0)
        else:
            v = v.fillna(prep["fill_values"][c])        # train median
        X[c] = (v - prep["means"][c]) / prep["sds"][c]  # standardise

    X["no_history"] = no_history.astype(float)
    X["odometer_missing"] = (df["prev_odometer"].isna() & ~no_history).astype(float)

    make = df["make"].fillna("UNKNOWN")
    for m in prep["makes"]:                    # top makes; everything else = reference
        X[f"make_{m}"] = (make == m).astype(float)

    return X[model["columns"]].to_numpy()


def raw_logit(df: pd.DataFrame, model: dict) -> np.ndarray:
    """The model's log-odds of failing, before calibration."""
    X = design_matrix(df, model)
    return model["intercept"] + X @ np.array(model["coefficients"])


def probability(df: pd.DataFrame, model: dict) -> np.ndarray:
    """Calibrated probability of failing: Platt scaling on the raw log-odds."""
    a, b = model["calibration"]["a"], model["calibration"]["b"]
    return 1.0 / (1.0 + np.exp(-(a * raw_logit(df, model) + b)))


def age_key(age_years: float) -> str:
    """Peer group for risk bands: whole years of age, with old cars grouped."""
    a = int(np.floor(age_years))
    return "25+" if a >= 25 else str(max(a, 3))


def band(df: pd.DataFrame, probs: np.ndarray, model: dict) -> np.ndarray:
    """Lower / Typical / Higher risk, against cars of the same age."""
    cuts = model["bands"]["cutoffs_by_age"]
    out = []
    for age, p in zip(df["age_years"], probs):
        lo, hi = cuts[age_key(age)]
        out.append("Lower risk" if p < lo else ("Typical" if p < hi else "Higher risk"))
    return np.array(out)