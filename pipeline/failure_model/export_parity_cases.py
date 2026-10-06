"""
Day 4, part 1: export parity test cases for the Worker.

Picks ~50 real training rows chosen to stress the tricky rules, rebuilds each
car's full MOT history from the database IN THE DVSA API's JSON SHAPE, and stores
what Python computed for it: the 7 numeric features, the make, the probability
and the risk band. worker/test/parity.test.mjs then runs the same histories
through the Worker's JavaScript and compares.

The mix:
  - cars with 2+ earlier fails            - first MOTs (no history)
  - unreadable/missing odometer           - long gaps between tests (450+ days)
  - cars with a retest within 60 days     - makes outside the top 20
  - random others

Run from the repository root:
    python -m pipeline.failure_model.export_parity_cases
Writes: worker/test/parity_cases.json
"""
import json

import numpy as np
import pandas as pd

from config import ROOT, TRAINING_ALL_CSV, VALIDATION_START
from pipeline.common import connect_db
from pipeline.failure_model.scoring import NUMERIC, band, probability

OUT = ROOT / "worker" / "test" / "parity_cases.json"
MODEL = ROOT / "models" / "model_v1.json"      # the artefact the Worker actually uses
SEED = 42
PER_GROUP = {"fails": 8, "no_history": 8, "no_odometer": 7, "long_gap": 7,
             "retest": 7, "rare_make": 6, "random": 7}


def vehicle_json(con, reg: str) -> dict:
    """One car's record, shaped like the DVSA API response the Worker receives."""
    make, model, first_used, registered = con.execute(
        "SELECT make, model, first_used_date, registration_date FROM vehicles "
        "WHERE registration = ?", (reg,)).fetchone()
    tests = []
    for num, done, result, odo, unit, odo_type, expiry, source in con.execute(
            "SELECT mot_test_number, completed_date, test_result, odometer_value, "
            "odometer_unit, odometer_result_type, expiry_date, data_source "
            "FROM mot_tests WHERE registration = ?", (reg,)):
        defects = [{"type": t, "text": x, "dangerous": bool(dg)} for t, x, dg in con.execute(
            "SELECT type, text, dangerous FROM defects WHERE mot_test_number = ?", (num,))]
        tests.append({
            "motTestNumber": num, "completedDate": done, "testResult": result,
            "odometerValue": None if odo is None else str(odo), "odometerUnit": unit,
            "odometerResultType": odo_type, "expiryDate": expiry,
            "dataSource": (source or "").upper(), "defects": defects,
        })
    return {"registration": reg, "make": make, "model": model,
            "firstUsedDate": first_used, "registrationDate": registered, "motTests": tests}


def has_retest(vehicle: dict) -> bool:
    days = sorted(pd.to_datetime([t["completedDate"][:10] for t in vehicle["motTests"]]))
    return any((b - a).days <= 60 for a, b in zip(days, days[1:]))


def main() -> None:
    model = json.loads(MODEL.read_text())
    df = pd.read_csv(TRAINING_ALL_CSV, dtype={"make": str})
    df = df[df["test_date"] < VALIDATION_START]          # any period works; keep it simple
    rng = np.random.default_rng(SEED)
    top_makes = set(model["preprocessing"]["makes"])

    pools = {
        "fails": df[df["n_prior_fails"] >= 2],
        "no_history": df[df["n_prior_fails"].isna()],
        "no_odometer": df[df["prev_odometer"].isna() & df["n_prior_fails"].notna()],
        "long_gap": df[df["days_since_prev"] > 450],
        "rare_make": df[~df["make"].isin(top_makes)],
        "random": df,
    }
    chosen, used = [], set()

    def take(group, frame, n):
        frame = frame[~frame["registration"].isin(used)]
        if len(frame) == 0:
            return
        picks = frame.iloc[rng.choice(len(frame), size=min(n, len(frame)), replace=False)]
        for _, row in picks.iterrows():
            used.add(row["registration"])
            chosen.append((group, row))

    con = connect_db()
    for g in ["fails", "no_history", "no_odometer", "long_gap", "rare_make"]:
        take(g, pools[g], PER_GROUP[g])

    # Retests need the raw history: look through a random batch of cars for some
    batch = df[~df["registration"].isin(used)]
    batch = batch.iloc[rng.choice(len(batch), size=min(400, len(batch)), replace=False)]
    found = 0
    for _, row in batch.iterrows():
        if found >= PER_GROUP["retest"]:
            break
        if row["registration"] not in used and has_retest(vehicle_json(con, row["registration"])):
            used.add(row["registration"])
            chosen.append(("retest", row))
            found += 1
    take("random", pools["random"], PER_GROUP["random"])

    rows = pd.DataFrame([r for _, r in chosen])
    probs = probability(rows, model)
    bands = band(rows, probs, model)

    cases = []
    for (group, row), p, b in zip(chosen, probs, bands):
        cases.append({
            "group": group,
            "target_date": row["test_date"],
            "vehicle": vehicle_json(con, row["registration"]),
            "expected": {
                "features": {c: (None if pd.isna(row[c]) else float(row[c])) for c in NUMERIC},
                "make": row["make"],
                "probability": float(p),
                "band": str(b),
            },
        })
    con.close()

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(cases, indent=1))
    counts = pd.Series([g for g, _ in chosen]).value_counts().to_dict()
    print(f"Wrote {len(cases)} cases to {OUT}")
    print("By group:", counts)


if __name__ == "__main__":
    main()