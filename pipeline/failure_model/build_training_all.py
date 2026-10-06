"""
All-vehicle model, step 1: build the training table.

Cars and light vans (see pipeline/vehicle_rules.py for what's excluded), a fixed
SAMPLE_PCT% of vehicles, and the eight features of the one-week roadmap:

  age_years, n_prior_fails, fails_last_3, prev_odometer, prev_fail_items,
  prev_advisories, days_since_prev, make
plus annual_miles_last_interval (swap candidate if fails_last_3 proves redundant)
and fuel_type (kept for segment checks, not a feature).

Same rules as the 3 Series build: GB tests only, targets from the 2018 rule change
to the snapshot end, retests within 60 days dropped, a fail wins a same-time tie,
and every history feature comes from EARLIER cycles only. No defect text is used.

Run from the repository root:
    python -m pipeline.failure_model.build_training_all
Output: data/processed/training_all.csv
"""
import csv
import hashlib
from collections import Counter

from config import (RETEST_GAP_DAYS, RULES_CHANGE, SAMPLE_PCT, SNAPSHOT_END,
                    TRAINING_ALL_CSV)
from pipeline.common import connect_db
from pipeline.vehicle_rules import clean_make, in_scope


def in_sample(registration) -> int:
    """Deterministic SAMPLE_PCT% sample by registration: the same cars every run,
    and each car's whole history stays together."""
    h = int(hashlib.md5((registration or "").encode()).hexdigest(), 16)
    return int(h % 100 < SAMPLE_PCT)


QUERY = """
-- 1. Vehicles: in the sample, in scope (cars and light vans), with a start date
WITH veh AS (
  SELECT registration,
         substr(COALESCE(first_used_date, registration_date),1,10) AS start_date,
         clean_make(make) AS make,
         UPPER(TRIM(fuel_type)) AS fuel_type
  FROM vehicles
  WHERE in_sample(registration) = 1
    AND in_scope(make, model) = 1
    AND COALESCE(first_used_date, registration_date) IS NOT NULL
),
-- 2. Their GB pass/fail tests up to the snapshot end
tests AS (
  SELECT m.mot_test_number, m.registration, m.completed_date,
         substr(m.completed_date,1,10) AS d,
         (m.test_result = 'FAILED') AS failed,
         CASE
           WHEN UPPER(COALESCE(m.odometer_result_type,'')) <> 'READ' THEN NULL
           WHEN LOWER(m.odometer_unit) LIKE 'k%' THEN m.odometer_value * 0.621371
           ELSE m.odometer_value
         END AS miles
  FROM veh v
  CROSS JOIN mot_tests m
  WHERE m.registration = v.registration
    AND UPPER(m.data_source) = 'DVSA'
    AND m.test_result IN ('PASSED','FAILED')
    AND substr(m.completed_date,1,10) <= :target_end
),
-- 3. Defect counts per test, by type only (no text)
items AS (
  SELECT d.mot_test_number,
         SUM(d.type = 'ADVISORY') AS n_advisory,
         SUM(d.type IN ('FAIL','MAJOR','DANGEROUS','PRS')) AS n_fail_items
  FROM tests t
  CROSS JOIN defects d
  WHERE d.mot_test_number = t.mot_test_number
  GROUP BY d.mot_test_number
),
-- 4. Attach counts; find each test's previous test date
ordered AS (
  SELECT t.*,
         COALESCE(i.n_advisory,0) AS n_advisory,
         COALESCE(i.n_fail_items,0) AS n_fail_items,
         LAG(t.d) OVER (PARTITION BY t.registration
                        ORDER BY t.completed_date, t.failed DESC) AS prev_any_d
  FROM tests t
  LEFT JOIN items i ON i.mot_test_number = t.mot_test_number
),
-- 5. Drop retests
cycles AS (
  SELECT * FROM ordered
  WHERE prev_any_d IS NULL OR julianday(d) - julianday(prev_any_d) > :gap
),
-- 6. Target and features, from earlier cycles only
feat AS (
  SELECT c.registration, c.mot_test_number, c.d AS test_date,
         c.failed AS target_failed,
         v.make, v.fuel_type,
         ROUND((julianday(c.d) - julianday(v.start_date)) / 365.25, 2) AS age_years,
         ROW_NUMBER() OVER w - 1 AS n_prior_cycles,
         SUM(c.failed) OVER (w ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING) AS n_prior_fails,
         SUM(c.failed) OVER (w ROWS BETWEEN 3 PRECEDING AND 1 PRECEDING) AS fails_last_3,
         LAG(c.miles, 1) OVER w AS prev_odometer,
         LAG(c.n_fail_items) OVER w AS prev_fail_items,
         LAG(c.n_advisory) OVER w AS prev_advisories,
         julianday(c.d) - julianday(LAG(c.d) OVER w) AS days_since_prev,
         LAG(c.miles, 1) OVER w - LAG(c.miles, 2) OVER w AS _miles_last_interval,
         julianday(LAG(c.d, 1) OVER w) - julianday(LAG(c.d, 2) OVER w) AS _days_last_interval
  FROM cycles c
  JOIN veh v ON v.registration = c.registration
  WINDOW w AS (PARTITION BY c.registration ORDER BY c.completed_date)
)
-- 7. Keep only the cycles we predict (history above was built from all years)
SELECT * FROM feat
WHERE test_date >= :target_start
ORDER BY registration, test_date
"""

# Safety net: makes still in scope with many small engines may be missed bike brands
SAFETY_QUERY = """
SELECT clean_make(make) AS make, COUNT(*) AS vehicles,
        COALESCE(SUM(engine_size < 900), 0) AS under_900cc
FROM vehicles v
WHERE in_sample(registration) = 1
  AND in_scope(make, model) = 1
  AND EXISTS (SELECT 1 FROM mot_tests m
              WHERE m.registration = v.registration
                AND UPPER(m.data_source) = 'DVSA'
                AND substr(m.completed_date,1,10) BETWEEN :target_start AND :target_end)
GROUP BY 1
HAVING COUNT(*) >= 20
"""
KNOWN_SMALL_CARS = {"SMART", "DAEWOO", "DACIA", "FIAT", "CHEVROLET", "PERODUA",
                    "SUZUKI", "DAIHATSU", "TOYOTA", "UNKNOWN"}


def main() -> None:
    con = connect_db()
    con.create_function("in_sample", 1, in_sample, deterministic=True)
    con.create_function("in_scope", 2, in_scope, deterministic=True)
    con.create_function("clean_make", 1, clean_make, deterministic=True)
    params = {"target_start": RULES_CHANGE, "target_end": SNAPSHOT_END, "gap": RETEST_GAP_DAYS}

    print(f"Building the all-vehicle table from a {SAMPLE_PCT}% sample of vehicles...")
    cur = con.execute(QUERY, params)
    cols = [c[0] for c in cur.description]
    rows = cur.fetchall()

    print("Running the motorcycle safety-net check...")
    safety = con.execute(SAFETY_QUERY, params).fetchall()
    con.close()

    # ---- Python step: annual mileage; drop the helper columns -----------------
    i = {c: k for k, c in enumerate(cols)}
    keep = [c for c in cols if not c.startswith("_")]
    out = []
    for r in rows:
        miles, days = r[i["_miles_last_interval"]], r[i["_days_last_interval"]]
        annual = round(miles / days * 365.25) if miles is not None and days and miles >= 0 else None
        out.append([r[i[c]] for c in keep] + [annual])
    keep.append("annual_miles_last_interval")
    assert out, "No rows - check the database and filters"

    TRAINING_ALL_CSV.parent.mkdir(parents=True, exist_ok=True)
    with TRAINING_ALL_CSV.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(keep)
        w.writerows(out)

    # ---- Sanity checks --------------------------------------------------------
    k = {c: n for n, c in enumerate(keep)}
    n = len(out)
    fails = sum(r[k["target_failed"]] for r in out)
    vehicles = len({r[k["registration"]] for r in out})
    print(f"\nWrote {TRAINING_ALL_CSV.name}: {n:,} test cycles from {vehicles:,} vehicles, "
          f"overall fail rate {100 * fails / n:.1f}%")
    no_hist = sum(r[k["n_prior_fails"]] is None for r in out)
    print(f"Cycles with no history (first test in the data): {no_hist:,} ({100 * no_hist / n:.1f}%)\n")

    makes = Counter(r[k["make"]] for r in out)
    print("Top 25 makes by test cycles:")
    for mk, c in makes.most_common(25):
        print(f"  {mk:<20} {c:>8,}  ({100 * c / n:4.1f}%)")

    print("\nSafety net: in-scope makes with over 20% of engines under 900cc")
    print("(anything not a known small car may be a missed motorcycle brand)")
    flagged = [(mk, v, u or 0) for mk, v, u in safety if v and (u or 0) / v > 0.2]
    for mk, v, u in sorted(flagged, key=lambda x: -x[1]):
        note = "known small cars" if mk in KNOWN_SMALL_CARS else "CHECK"
        print(f"  {mk:<20} {v:>6,} vehicles, {100 * u / v:4.1f}% under 900cc  {note}")
    if not flagged:
        print("  none")


if __name__ == "__main__":
    main()