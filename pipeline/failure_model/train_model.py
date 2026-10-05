"""
Failure model, step 2: train and evaluate (BMW 3 Series pilot).

Each model is trained with and without the advisory-category features, so the
gain from the text is measured directly.

  1. Baseline            - fail rate by generation and age (the chart)
  2. Logistic            - original features
  3. Boosting            - original features
  4. Logistic + text     - original + advisory categories
  5. Boosting + text     - original + advisory categories

Three-way split by date (roadmap Phase 0):
  train       tests before VALIDATION_START       -> fit models
  validation  VALIDATION_START to TEST_START      -> every score printed here
  test        from TEST_START                     -> NOT scored; held back for Phase 7

Run from the repository root:
    python -m pipeline.failure_model.train_model
Reads data/processed/training_3series.csv; saves plots and the reference
scores to outputs/.
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from config import (MIN_TESTS_PER_POINT, OUTPUTS, TEST_START, TRAINING_CSV,
                    VALIDATION_START)
from pipeline.common import generation

DATA = TRAINING_CSV
MIN_CELL = MIN_TESTS_PER_POINT

BASE_NUMERIC = [
    "age_years", "reg_year", "engine_size",
    "n_prior_cycles", "n_prior_fails", "prev_failed", "prev_advisories",
    "prev_fail_items", "prev_minors", "days_since_prev",
]
# Removed: odometer_miles, miles_since_prev, miles_per_year.
# All three used the odometer reading taken AT the test being predicted,
# which the live site won't have before the test. The columns are still in
# the CSV; they're just no longer given to the models.
TEXT_FEATURES = [
    "prev_adv_tyres", "prev_adv_brakes", "prev_adv_suspension", "prev_adv_steering",
    "prev_adv_leaks", "prev_adv_other", "prev_adv_corrosion", "prev_adv_tyre_limit",
]
MAIN_FUELS = ["DIESEL", "PETROL", "HYBRID ELECTRIC (CLEAN)", "ELECTRIC DIESEL"]
FUEL_COLS = [f"fuel_{f}" for f in MAIN_FUELS + ["OTHER"]]


def prepare(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["gen"] = df["reg_year"].apply(generation)
    df["age_int"] = df["age_years"].round().astype(int)
    fuel = df["fuel_type"].where(df["fuel_type"].isin(MAIN_FUELS), "OTHER")
    for f in MAIN_FUELS + ["OTHER"]:
        df[f"fuel_{f}"] = (fuel == f).astype(int)
    return df


def baseline_predict(train: pd.DataFrame, test: pd.DataFrame) -> np.ndarray:
    """The chart as a model: fail rate for this generation at this age."""
    cell = (train.groupby(["gen", "age_int"])["target_failed"]
                 .agg(["mean", "count"]).reset_index())
    cell = cell[cell["count"] >= MIN_CELL][["gen", "age_int", "mean"]]
    gen_mean = train.groupby("gen")["target_failed"].mean().rename("gen_mean").reset_index()
    out = (test[["gen", "age_int"]]
           .merge(cell, on=["gen", "age_int"], how="left")
           .merge(gen_mean, on="gen", how="left"))
    pred = out["mean"].fillna(out["gen_mean"]).fillna(train["target_failed"].mean())
    return pred.to_numpy()


def make_logit():
    return make_pipeline(
        SimpleImputer(strategy="median", add_indicator=True),
        StandardScaler(),
        LogisticRegression(max_iter=2000),
    )


def make_gbm():
    return HistGradientBoostingClassifier(
        learning_rate=0.05, max_iter=500, max_leaf_nodes=31,
        l2_regularization=1.0, early_stopping=True,
        validation_fraction=0.15, random_state=0,
    )


def score(name: str, y: np.ndarray, p: np.ndarray, base_rate: float) -> dict:
    brier = brier_score_loss(y, p)
    brier_ref = brier_score_loss(y, np.full_like(p, base_rate))
    return {
        "model": name,
        "AUC": round(roc_auc_score(y, p), 3),
        "Brier": round(brier, 4),
        "Brier skill": round(1 - brier / brier_ref, 3),
        "Log loss": round(log_loss(y, p), 4),
    }


def main() -> None:
    df = prepare(pd.read_csv(DATA))
    base_features = BASE_NUMERIC + FUEL_COLS
    text_features = base_features + TEXT_FEATURES

    train = df[df["test_date"] < VALIDATION_START]
    # From here on, "test" in variable names means the VALIDATION period.
    # The real test set (from TEST_START) is deliberately never touched.
    test = df[(df["test_date"] >= VALIDATION_START) & (df["test_date"] < TEST_START)]
    n_held_back = int((df["test_date"] >= TEST_START).sum())
    y_train, y_test = train["target_failed"].to_numpy(), test["target_failed"].to_numpy()
    base_rate = y_train.mean()

    print(f"Train:      {len(train):,} cycles before {VALIDATION_START}, fail rate {100*y_train.mean():.1f}%")
    print(f"Validation: {len(test):,} cycles from {VALIDATION_START} to {TEST_START}, "
          f"fail rate {100*y_test.mean():.1f}%")
    print(f"Test:       {n_held_back:,} cycles from {TEST_START} - held back for Phase 7, not scored\n")

    preds = {"Baseline (gen + age)": baseline_predict(train, test)}
    fitted = {}
    for label, feats in [("", base_features), (" + text", text_features)]:
        for name, factory in [("Logistic", make_logit), ("Boosting", make_gbm)]:
            model = factory().fit(train[feats], y_train)
            preds[name + label] = model.predict_proba(test[feats])[:, 1]
            fitted[name + label] = (model, feats)

    results = pd.DataFrame([score(k, y_test, p, base_rate) for k, p in preds.items()])
    print("Validation-set performance:")
    print(results.to_string(index=False), "\n")

    # Save the reference scores, to copy into docs/decisions.md
    OUTPUTS.mkdir(parents=True, exist_ok=True)
    results.to_csv(OUTPUTS / "reference_scores.csv", index=False)
    print("Saved outputs/reference_scores.csv\n")

    # What does the boosting model rely on once the text features are in?
    gbm, feats = fitted["Boosting + text"]
    imp = permutation_importance(gbm, test[feats], y_test, scoring="roc_auc",
                                 n_repeats=5, random_state=0)
    ranked = (pd.Series(imp["importances_mean"], index=feats)
                .sort_values(ascending=False).round(4))
    print("Permutation importance, Boosting + text (drop in validation AUC when shuffled):")
    print(ranked.head(15).to_string(), "\n")

    # Logistic coefficients for the text features: direction and size of each effect
    logit, feats = fitted["Logistic + text"]
    coefs = pd.Series(logit[-1].coef_[0][:len(feats)], index=feats)
    print("Logistic + text: standardised coefficients for advisory categories")
    print("(positive = raises fail risk; per one standard deviation of the feature)")
    print(coefs[TEXT_FEATURES].sort_values(ascending=False).round(3).to_string(), "\n")

    # Calibration plot
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.plot([0, 1], [0, 1], linestyle="--", color="grey", label="Perfect calibration")
    for name in ["Baseline (gen + age)", "Logistic + text", "Boosting + text"]:
        frac, mean_pred = calibration_curve(y_test, preds[name], n_bins=10, strategy="quantile")
        ax.plot(mean_pred, frac, marker="o", label=name)
    ax.set_xlabel("Predicted fail probability")
    ax.set_ylabel("Actual fail rate")
    ax.set_title(f"Calibration on validation ({VALIDATION_START} to {TEST_START})")
    ax.set_xlim(0, 0.8); ax.set_ylim(0, 0.8)
    ax.legend(); ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUTPUTS / "calibration_3series.png", dpi=150)
    print("Saved calibration_3series.png\n")

    # Relative-risk bands, before and after text features
    for name in ["Logistic", "Logistic + text"]:
        bands = test.copy()
        bands["p"] = preds[name]
        bands["peer_pct"] = bands.groupby("age_int")["p"].rank(pct=True) * 100
        bands["band"] = pd.cut(bands["peer_pct"], [0, 40, 80, 100],
                               labels=["Lower risk", "Typical", "Higher risk"])
        print(f"Actual fail rate by relative-risk band ({name}):")
        print(bands.groupby("band", observed=True)["target_failed"]
                   .agg(fail_rate="mean", cars="count")
                   .assign(fail_rate=lambda d: (d["fail_rate"] * 100).round(1))
                   .to_string(), "\n")


if __name__ == "__main__":
    main()
