"""
All-vehicle model, step 3: the ONE run on the test set.

Scores the 2024 to February 2026 test period with outputs/model.json, through
scoring.py - exactly the maths the Worker will use - and checks every gate:

  Discrimination  AUC with a bootstrap 95% interval, against an age-only baseline
  Calibration     slope and intercept, Brier skill, calibration plot
  Risk bands      fail rate must rise from Lower to Typical to Higher risk
  Segments        make, fuel, age band, history vs none: AUC >= 0.58 and
                  calibration within 5 points
  Stability       PSI of the model score, train vs test

Run it once, after train_final.py, and don't go back to tune: a failed gate is
reported as a limitation.

Run from the repository root:
    python -m pipeline.failure_model.validate
Writes to outputs/validation/: report.txt, segments.csv, calibration.png
"""
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, roc_auc_score

from config import OUTPUTS, TEST_START, TRAINING_ALL_CSV, VALIDATION_START
from pipeline.failure_model.scoring import age_key, band, probability, raw_logit

TARGET = "target_failed"
N_BOOT = 200
SEGMENT_MIN = 500          # segments smaller than this are reported but not gated
GATE_AUC = 0.58
GATE_CAL_POINTS = 5.0


def psi(expected: np.ndarray, actual: np.ndarray, bins: int = 10) -> float:
    edges = np.unique(np.quantile(expected, np.linspace(0, 1, bins + 1)))
    edges[0], edges[-1] = -np.inf, np.inf
    e = np.histogram(expected, edges)[0] / len(expected) + 1e-6
    a = np.histogram(actual, edges)[0] / len(actual) + 1e-6
    return float(((a - e) * np.log(a / e)).sum())


def main() -> None:
    model = json.loads((OUTPUTS / "model.json").read_text())
    df = pd.read_csv(TRAINING_ALL_CSV, dtype={"make": str})
    train = df[df["test_date"] < VALIDATION_START]
    test = df[df["test_date"] >= TEST_START].copy()
    y = test[TARGET].to_numpy()
    lines = []

    def say(text=""):
        print(text)
        lines.append(text)

    say(f"TEST SET VALIDATION  ({len(test):,} cycles from {TEST_START}, fail rate {100 * y.mean():.1f}%)")
    say(f"Model: {model['name']} v{model['version']}, built {model['built']}\n")

    p = probability(test, model)
    test["p"] = p
    base_rate = float(train[TARGET].mean())
    gates = []

    # ---- Discrimination ---------------------------------------------------------
    age_rates = train.groupby(train["age_years"].apply(age_key))[TARGET].mean()
    p_base = test["age_years"].apply(age_key).map(age_rates).fillna(base_rate).to_numpy()
    auc, auc_base = roc_auc_score(y, p), roc_auc_score(y, p_base)
    rng = np.random.default_rng(0)
    boots, boots_base = [], []
    for _ in range(N_BOOT):
        idx = rng.integers(0, len(y), len(y))
        boots.append(roc_auc_score(y[idx], p[idx]))
        boots_base.append(roc_auc_score(y[idx], p_base[idx]))
    lo, hi = np.percentile(boots, [2.5, 97.5])
    blo, bhi = np.percentile(boots_base, [2.5, 97.5])
    say("Discrimination")
    say(f"  Model AUC      {auc:.3f}  (95% CI {lo:.3f} to {hi:.3f}),  Gini {2 * auc - 1:.3f}")
    say(f"  Baseline AUC   {auc_base:.3f}  (95% CI {blo:.3f} to {bhi:.3f})  age only")
    gates.append(("Beats the baseline, intervals not overlapping", lo > bhi))

    # ---- Calibration -------------------------------------------------------------
    z = model["calibration"]["a"] * raw_logit(test, model) + model["calibration"]["b"]
    fit = LogisticRegression(C=1e6, max_iter=1000).fit(z.reshape(-1, 1), y)
    slope, intercept = float(fit.coef_[0][0]), float(fit.intercept_[0])
    skill = 1 - brier_score_loss(y, p) / brier_score_loss(y, np.full(len(y), base_rate))
    gap = 100 * (p.mean() - y.mean())
    say("\nCalibration")
    say(f"  Slope {slope:.3f} (gate 0.9 to 1.1), intercept {intercept:+.3f}")
    say(f"  Mean predicted {100 * p.mean():.1f}% vs actual {100 * y.mean():.1f}%  (gap {gap:+.1f} points)")
    say(f"  Brier skill {skill:.3f}")
    gates.append(("Calibration slope 0.9 to 1.1", 0.9 <= slope <= 1.1))

    deciles = pd.qcut(p, 10, labels=False, duplicates="drop")
    cal = pd.DataFrame({"p": p, "y": y, "d": deciles}).groupby("d").mean()
    worst_bin = float((cal["p"] - cal["y"]).abs().max() * 100)
    say(f"  Worst decile off by {worst_bin:.1f} points")
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.plot([0, 1], [0, 1], "--", color="grey", label="Perfect calibration")
    ax.plot(cal["p"], cal["y"], "o-", label="Model, test set")
    top = float(max(cal["p"].max(), cal["y"].max())) * 1.1
    ax.set_xlim(0, top); ax.set_ylim(0, top)
    ax.set_xlabel("Predicted fail probability"); ax.set_ylabel("Actual fail rate")
    ax.set_title(f"Calibration on the test set (from {TEST_START})")
    ax.legend(); ax.grid(alpha=0.3); fig.tight_layout()

    # ---- Risk bands ----------------------------------------------------------------
    test["band"] = band(test, p, model)
    say("\nRisk bands")
    rates = []
    for b_name in ["Lower risk", "Typical", "Higher risk"]:
        m = test["band"] == b_name
        rates.append(y[m.to_numpy()].mean())
        say(f"  {b_name:<12} {100 * rates[-1]:5.1f}% fail  ({m.sum():,} cars)")
    gates.append(("Bands rise from Lower to Typical to Higher risk", rates[0] < rates[1] < rates[2]))

    # ---- Segments ------------------------------------------------------------------
    makes = set(model["preprocessing"]["makes"])
    test["make_seg"] = test["make"].where(test["make"].isin(makes), "OTHER MAKES")
    test["age_seg"] = pd.cut(test["age_years"], [0, 5, 8, 11, 15, 100],
                             labels=["3-5", "5-8", "8-11", "11-15", "15+"]).astype(str)
    test["history_seg"] = np.where(test["n_prior_fails"].isna(), "no history", "has history")
    seg_rows = []
    for dim, col in [("make", "make_seg"), ("fuel", "fuel_type"),
                     ("age band", "age_seg"), ("history", "history_seg")]:
        for name, g in test.groupby(col):
            if g[TARGET].nunique() < 2:
                continue
            s_auc = roc_auc_score(g[TARGET], g["p"])
            s_gap = 100 * (g["p"].mean() - g[TARGET].mean())
            gated = len(g) >= SEGMENT_MIN
            ok = (s_auc >= GATE_AUC and abs(s_gap) <= GATE_CAL_POINTS) if gated else None
            seg_rows.append({"dimension": dim, "segment": name, "cars": len(g),
                             "fail_rate": round(g[TARGET].mean(), 4), "auc": round(s_auc, 3),
                             "calibration_gap_points": round(s_gap, 1),
                             "result": "pass" if ok else ("FAIL" if ok is False else "too small")})
    seg = pd.DataFrame(seg_rows)
    failed = seg[seg["result"] == "FAIL"]
    say(f"\nSegments ({len(seg)} checked; gate: AUC >= {GATE_AUC}, calibration within "
        f"{GATE_CAL_POINTS:.0f} points, segments of {SEGMENT_MIN}+ cars)")
    if len(failed):
        for _, r in failed.iterrows():
            say(f"  FAIL  {r['dimension']}: {r['segment']}  AUC {r['auc']}, "
                f"gap {r['calibration_gap_points']:+.1f} points, {r['cars']:,} cars")
    else:
        say("  All gated segments pass")
    gates.append(("No segment fails AUC or calibration", len(failed) == 0))

    # ---- Stability -----------------------------------------------------------------
    score_psi = psi(probability(train, model), p)
    say(f"\nStability: score PSI train vs test {score_psi:.3f} "
        f"({'stable' if score_psi < 0.1 else 'monitor' if score_psi < 0.25 else 'unstable'})")
    gates.append(("Score PSI below 0.25", score_psi < 0.25))

    # ---- Verdict -------------------------------------------------------------------
    say("\nGATES")
    for name, ok in gates:
        say(f"  {'PASS' if ok else 'FAIL'}  {name}")
    say("\nA failed gate is reported as a limitation, not tuned away.")

    out = OUTPUTS / "validation"
    out.mkdir(parents=True, exist_ok=True)
    (out / "report.txt").write_text("\n".join(lines))
    seg.to_csv(out / "segments.csv", index=False)
    fig.savefig(out / "calibration.png", dpi=150)
    print(f"\nSaved report.txt, segments.csv and calibration.png to {out}")


if __name__ == "__main__":
    main()