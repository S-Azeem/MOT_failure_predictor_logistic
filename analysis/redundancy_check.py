"""
Redundancy check for the failure model's features (roadmap Phase 3).

1. Correlation matrix (Spearman): how strongly each pair of features moves together.
2. VIF (variance inflation factor): how well each feature can be predicted
   from ALL the others combined - catches redundancy no single pair shows.

Run from the repository root:
    python -m analysis.redundancy_check
"""
import pandas as pd
from sklearn.linear_model import LinearRegression

from config import TRAINING_CSV

FEATURES = [
    "age_years", "reg_year", "engine_size",
    "n_prior_cycles", "n_prior_fails", "prev_failed", "prev_advisories",
    "prev_fail_items", "prev_minors", "days_since_prev",
    "prev_adv_tyres", "prev_adv_brakes", "prev_adv_suspension", "prev_adv_steering",
    "prev_adv_leaks", "prev_adv_other", "prev_adv_corrosion", "prev_adv_tyre_limit",
]
# Fuel dummies are left out on purpose: one-hot columns always sum to 1, so
# their VIF is infinite by construction (the "dummy variable trap").

df = pd.read_csv(TRAINING_CSV)

# Rows with history only: history features are blank for a car's first test.
X = df[FEATURES].dropna()
print(f"Rows with complete features: {len(X):,} of {len(df):,}\n")

# ---- 1. Pairwise correlation (Spearman: rank-based, suits skewed counts) ----
corr = X.corr(method="spearman")
pd.set_option("display.width", 250)
print("Spearman correlation matrix:")
print(corr.round(2).to_string(), "\n")

pairs = []
for a_idx, a in enumerate(FEATURES):
    for b in FEATURES[a_idx + 1:]:
        pairs.append((a, b, corr.loc[a, b]))
strong = sorted((p for p in pairs if abs(p[2]) >= 0.5), key=lambda p: -abs(p[2]))
print("Pairs with |r| >= 0.5 (rule of thumb: above 0.7 is strong redundancy):")
for a, b, r in strong:
    print(f"  {a:>20} ~ {b:<20} r = {r:+.2f}")
if not strong:
    print("  none")
print()

# ---- 2. Variance inflation factor -------------------------------------------
# Regress each feature on all the others: VIF = 1 / (1 - R^2).
# 1 = unique information; above 5 = worth a look; above 10 = largely redundant.
print("Variance inflation factors:")
vifs = {}
for f in FEATURES:
    others = [c for c in FEATURES if c != f]
    r2 = LinearRegression().fit(X[others], X[f]).score(X[others], X[f])
    vifs[f] = float("inf") if r2 >= 1 else 1 / (1 - r2)
for f, v in sorted(vifs.items(), key=lambda kv: -kv[1]):
    flag = "  <- high" if v > 10 else ("  <- check" if v > 5 else "")
    print(f"  {f:>20}: {v:6.1f}{flag}")
