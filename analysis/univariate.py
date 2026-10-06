"""
Phase 2: univariate analysis - every candidate feature, one at a time.

For each feature, using TRAIN data for the statistics and VALIDATION for stability:
  - missing rate, min, max, number of distinct values
  - bins (deciles for continuous features, one bin per value for small counts,
    and "missing" as its own bin) and the fail rate in each bin
  - weight of evidence (WoE) per bin and information value (IV)
  - univariate AUC: how well the feature alone ranks fails above passes
  - monotonicity: does the fail rate move one way across the bins?
  - PSI: has the feature's distribution shifted from train to validation?
The test period (from TEST_START) is not touched.

Run from the repository root:
    python -m analysis.univariate
Writes to outputs/:
    univariate_summary.csv    one row per feature, sorted by IV, with gate flags
    univariate_bins.csv       every bin: counts, fail rates, WoE (train and validation)
    univariate_by_year.csv    fail rate per bin per test year (is the pattern stable?)
"""
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from config import OUTPUTS, TEST_START, TRAINING_CSV, VALIDATION_START

NUMERIC = [
    # the car
    "age_years", "reg_year", "engine_size",
    # track record
    "n_prior_cycles", "n_prior_fails", "fails_last_3",
    "n_prior_fails_pre2018", "n_prior_fails_post2018", "prev_test_pre2018",
    # previous test
    "prev_failed", "prev_fail_items", "prev_advisories", "prev_minors",
    "prev_adv_tyres", "prev_adv_brakes", "prev_adv_suspension", "prev_adv_steering",
    "prev_adv_leaks", "prev_adv_other", "prev_adv_corrosion", "prev_adv_tyre_limit",
    "advisory_trend", "n_recurring_categories",
    # use and mileage
    "days_since_prev", "prev_odometer", "annual_miles_last_interval", "projected_miles",
]
CATEGORICAL = ["fuel_type"]

N_BINS = 10            # deciles for continuous features
MAX_DISCRETE = 12      # a feature with this many distinct values or fewer gets one bin per value
MIN_BIN_SHARE = 0.02   # rare high values of a count are grouped into one top bin ("4+")
SMOOTH = 0.5           # added to bin counts so WoE never divides by zero
MONO_MIN_N = 100       # bins smaller than this are ignored when judging monotonicity
TARGET = "target_failed"


def fmt(v) -> str:
    """1.0 -> '1', 2.5 -> '2.5'."""
    return str(int(v)) if float(v).is_integer() else str(v)


def make_binner(train_values: pd.Series, categorical: bool):
    """Learn bin rules from TRAIN data; return a function that bins any series the same way."""
    present = train_values.dropna()
    if categorical:
        known = set(present.unique())

        def bin_category(s: pd.Series) -> pd.Series:
            def label(v):
                if pd.isna(v):
                    return "missing"
                return str(v) if v in known else "other (unseen in train)"
            return s.map(label)
        return bin_category, False

    if present.nunique() <= MAX_DISCRETE or (present % 1 == 0).all() and present.nunique() <= 40:
        # A count or a small set of values: one bin per value, with the rare upper
        # tail grouped so the top bin holds at least MIN_BIN_SHARE of the rows.
        counts = present.value_counts().sort_index()
        tail = counts[::-1].cumsum()[::-1]                    # rows at or above each value
        big_enough = tail[tail >= MIN_BIN_SHARE * len(present)]
        cap = big_enough.index.max() if len(big_enough) else counts.index.max()
        has_tail = (present > cap).any()

        def bin_count(s: pd.Series) -> pd.Series:
            def label(v):
                if pd.isna(v):
                    return "missing"
                if has_tail and v >= cap:
                    return f"{fmt(cap)}+"
                return fmt(v)
            return s.map(label)
        return bin_count, False

    # Continuous: decile edges from train; open-ended first and last bins
    edges = np.unique(np.nanquantile(present, np.linspace(0, 1, N_BINS + 1)))
    edges = np.concatenate([[-np.inf], edges[1:-1], [np.inf]])

    def bin_continuous(s: pd.Series) -> pd.Series:
        binned = pd.cut(s, edges, include_lowest=True).astype(str)
        return binned.where(s.notna(), "missing")
    return bin_continuous, True


def bin_order(labels, ordered_values):
    """Sort bins in their natural order: by value, with 'missing' last."""
    return sorted(labels, key=lambda b: (b in ("missing", "other (unseen in train)"),
                                         ordered_values.get(b, 0)))


def woe_table(bins: pd.Series, y: pd.Series) -> pd.DataFrame:
    """Counts, fail rate and weight of evidence per bin.
    WoE = ln(share of all passes in the bin / share of all fails in the bin).
    Negative WoE = riskier than average; positive = safer."""
    t = pd.DataFrame({"bin": bins, "y": y}).groupby("bin")["y"].agg(n="count", fails="sum")
    t["passes"] = t["n"] - t["fails"]
    pass_share = (t["passes"] + SMOOTH) / (t["passes"].sum() + SMOOTH * len(t))
    fail_share = (t["fails"] + SMOOTH) / (t["fails"].sum() + SMOOTH * len(t))
    t["fail_rate"] = t["fails"] / t["n"]
    t["woe"] = np.log(pass_share / fail_share)
    t["iv_part"] = (pass_share - fail_share) * t["woe"]
    return t


def psi(train_bins: pd.Series, val_bins: pd.Series) -> float:
    """Population stability index between two binned distributions."""
    labels = sorted(set(train_bins) | set(val_bins))
    p = train_bins.value_counts(normalize=True).reindex(labels, fill_value=0) + 1e-6
    q = val_bins.value_counts(normalize=True).reindex(labels, fill_value=0) + 1e-6
    return float(((q - p) * np.log(q / p)).sum())


def monotonic(rates: list, sizes: list) -> str:
    """Direction of the fail rate across ordered bins (missing excluded).
    Uses only bins with at least MONO_MIN_N rows, so tiny noisy bins can't flip it,
    and the Spearman rank correlation between bin order and fail rate:
    +0.8 or more = rising, -0.8 or less = falling, otherwise mixed."""
    kept = [r for r, n in zip(rates, sizes) if n >= MONO_MIN_N]
    if len(kept) < 3:
        return "too few bins"
    rho = pd.Series(kept).corr(pd.Series(range(len(kept))), method="spearman")
    if rho >= 0.8:
        return f"rising ({rho:.2f})"
    if rho <= -0.8:
        return f"falling ({rho:.2f})"
    return f"mixed ({rho:.2f})"


def main() -> None:
    df = pd.read_csv(TRAINING_CSV)
    missing_cols = [c for c in NUMERIC + CATEGORICAL if c not in df.columns]
    if missing_cols:
        raise SystemExit(f"Columns not in the training table (rebuild it?): {missing_cols}")

    train = df[df["test_date"] < VALIDATION_START].copy()
    val = df[(df["test_date"] >= VALIDATION_START) & (df["test_date"] < TEST_START)].copy()
    train["year"] = train["test_date"].str[:4]
    print(f"Train: {len(train):,} cycles   Validation: {len(val):,} cycles   (test period not used)\n")

    summary, bin_rows, year_rows = [], [], []
    for feat in NUMERIC + CATEGORICAL:
        is_cat = feat in CATEGORICAL
        binner, continuous = make_binner(train[feat], is_cat)
        tb, vb = binner(train[feat]), binner(val[feat])

        t_tab = woe_table(tb, train[TARGET])
        v_tab = val.groupby(vb)[TARGET].agg(n_val="count", fail_rate_val="mean")

        # Natural bin order: by the lowest value falling in each bin
        order_key = (train.groupby(tb)[feat].min().to_dict() if not is_cat
                     else {b: k for k, b in enumerate(t_tab.sort_values("fail_rate").index)})
        ordered = bin_order(list(t_tab.index), order_key)
        t_tab = t_tab.loc[ordered]

        # Univariate AUC: score each row by its bin's TRAIN fail rate
        rate_map = t_tab["fail_rate"].to_dict()
        auc_train = roc_auc_score(train[TARGET], tb.map(rate_map))
        val_scores = vb.map(rate_map).fillna(train[TARGET].mean())
        auc_val = roc_auc_score(val[TARGET], val_scores)

        real_bins = [b for b in ordered if b not in ("missing", "other (unseen in train)")]
        iv = float(t_tab["iv_part"].sum())
        p = psi(tb, vb)
        mono = ("n/a (category)" if is_cat else
                monotonic(t_tab.loc[real_bins, "fail_rate"].tolist(), t_tab.loc[real_bins, "n"].tolist()))

        flags = []
        if iv < 0.02:
            flags.append("IV<0.02: drop unless domain reason")
        if iv > 0.5:
            flags.append("IV>0.5: check for leakage")
        if p > 0.25:
            flags.append("PSI>0.25: unstable")
        elif p > 0.1:
            flags.append("PSI 0.1-0.25: monitor")
        if mono.startswith("mixed"):
            flags.append("non-monotonic: explain or rebin")

        summary.append({
            "feature": feat,
            "missing_pct": round(100 * train[feat].isna().mean(), 1),
            "min": None if is_cat else train[feat].min(),
            "max": None if is_cat else train[feat].max(),
            "distinct": train[feat].nunique(),
            "bins": len(t_tab),
            "IV": round(iv, 4),
            "AUC_train": round(auc_train, 3),
            "AUC_val": round(auc_val, 3),
            "PSI": round(p, 4),
            "monotonic": mono,
            "flags": "; ".join(flags),
        })

        merged = t_tab.join(v_tab, how="left")
        for b, r in merged.iterrows():
            bin_rows.append({
                "feature": feat, "bin": b, "n_train": int(r["n"]),
                "fail_rate_train": round(r["fail_rate"], 4), "woe": round(r["woe"], 4),
                "n_val": None if pd.isna(r["n_val"]) else int(r["n_val"]),
                "fail_rate_val": None if pd.isna(r["fail_rate_val"]) else round(r["fail_rate_val"], 4),
            })

        by_year = train.groupby([tb, "year"])[TARGET].agg(n="count", fail_rate="mean").reset_index()
        by_year.columns = ["bin", "year", "n", "fail_rate"]
        by_year.insert(0, "feature", feat)
        year_rows.append(by_year)

    summary_df = pd.DataFrame(summary).sort_values("IV", ascending=False)
    OUTPUTS.mkdir(parents=True, exist_ok=True)
    summary_df.to_csv(OUTPUTS / "univariate_summary.csv", index=False)
    pd.DataFrame(bin_rows).to_csv(OUTPUTS / "univariate_bins.csv", index=False)
    pd.concat(year_rows).round(4).to_csv(OUTPUTS / "univariate_by_year.csv", index=False)

    pd.set_option("display.width", 250)
    pd.set_option("display.max_colwidth", 60)
    print(summary_df.drop(columns=["min", "max"]).to_string(index=False))
    print("\nIV guide: <0.02 useless, 0.02-0.1 weak, 0.1-0.3 medium, >0.3 strong, >0.5 check leakage")
    print("PSI guide: <0.1 stable, 0.1-0.25 monitor, >0.25 unstable")
    print("\nSaved outputs/univariate_summary.csv, univariate_bins.csv, univariate_by_year.csv")


if __name__ == "__main__":
    main()
