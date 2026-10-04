"""
Website data: BMW 3 Series MOT fail rate by age and generation.

Reads the local 1% SQLite sample and writes a small JSON file into the
website repository, where failure-model.html draws the chart from it.

Run from the repository root:
    python -m pipeline.site_data.build_chart_data
Then commit and push the JSON from the website repository.
"""
import json
import math
from datetime import date

from config import CHART_JSON, MIN_TESTS_PER_POINT, RULES_CHANGE, SNAPSHOT_END
from pipeline.common import connect_db

OUT_PATH = CHART_JSON
WINDOW_START = RULES_CHANGE   # post-2018 defect rules only
WINDOW_END = SNAPSHOT_END     # bulk snapshot end; later tests are API-refreshed cars only
MIN_TESTS = MIN_TESTS_PER_POINT
MIN_AGE = 3

QUERY = """
WITH v AS (
  SELECT registration,
         substr(COALESCE(first_used_date, registration_date),1,10) AS start_date
  FROM vehicles
  WHERE make = 'BMW'
    AND (model = '3 SERIES' OR model GLOB '3[0-9][0-9]*')
),
t AS (
  SELECT m.registration,
         m.completed_date,
         (m.test_result = 'FAILED') AS failed,
         CAST(substr(v.start_date,1,4) AS INT) AS reg_year,
         CAST((julianday(substr(m.completed_date,1,10))
             - julianday(v.start_date)) / 365.25 + 0.5 AS INT) AS age
  FROM v
  CROSS JOIN mot_tests m
  WHERE m.registration = v.registration
    AND v.start_date IS NOT NULL
    AND UPPER(m.data_source) = 'DVSA'
    AND m.test_result IN ('PASSED', 'FAILED')
    AND substr(m.completed_date,1,10) BETWEEN :start AND :end
),
g AS (
  SELECT *,
    CASE
      WHEN reg_year <  1991 THEN 'E30 and older'
      WHEN reg_year <= 1998 THEN 'E36'
      WHEN reg_year <= 2005 THEN 'E46'
      WHEN reg_year <= 2012 THEN 'E90'
      WHEN reg_year <= 2019 THEN 'F30'
      ELSE 'G20'
    END AS gen,
    ROW_NUMBER() OVER (
      PARTITION BY registration, age
      ORDER BY completed_date, failed DESC   -- a fail wins a same-timestamp tie
    ) AS rn
  FROM t
)
SELECT gen, age, COUNT(*) AS tests, SUM(failed) AS fails
FROM g
WHERE rn = 1 AND age >= :min_age
GROUP BY gen, age
HAVING COUNT(*) >= :min_tests
ORDER BY gen, age
"""

GEN_ORDER = ["E30 and older", "E36", "E46", "E90", "F30", "G20"]


def wilson(fails: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """95% Wilson score interval for a proportion, returned in percent."""
    p = fails / n
    denom = 1 + z**2 / n
    centre = (p + z**2 / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / denom
    return round(100 * (centre - half), 1), round(100 * (centre + half), 1)


def main() -> None:
    con = connect_db()
    rows = con.execute(QUERY, {
        "start": WINDOW_START, "end": WINDOW_END,
        "min_age": MIN_AGE, "min_tests": MIN_TESTS,
    }).fetchall()
    con.close()

    series = {}
    for gen, age, tests, fails in rows:
        lo, hi = wilson(fails, tests)
        series.setdefault(gen, []).append({
            "age": age, "tests": tests, "fails": fails,
            "fail_pct": round(100 * fails / tests, 1),
            "ci_low": lo, "ci_high": hi,
        })

    # ---- Sanity checks: fail loudly rather than ship a broken chart ----------
    assert series, "Query returned nothing - check the database and filters"
    for gen, points in series.items():
        assert gen in GEN_ORDER, f"Unexpected generation label: {gen}"
        for p in points:
            assert 0 <= p["fail_pct"] <= 100, f"Bad rate in {gen}: {p}"
            assert p["tests"] >= MIN_TESTS, f"Small cell leaked through: {gen} {p}"
            assert p["ci_low"] <= p["fail_pct"] <= p["ci_high"], f"Bad CI: {gen} {p}"
    total_tests = sum(p["tests"] for pts in series.values() for p in pts)
    assert total_tests > 10_000, f"Suspiciously few tests: {total_tests}"
    # --------------------------------------------------------------------------

    output = {
        "meta": {
            "title": "BMW 3 Series MOT fail rate by age and generation",
            "built": date.today().isoformat(),
            "source": "DVSA MOT history, 1% vehicle sample",
            "population": "Great Britain tests only (DVSA); vehicles with at least one test in the window",
            "test_window": [WINDOW_START, WINDOW_END],
            "fail_definition": "Test recorded as failed, including faults rectified during the test (PRS)",
            "counting": "First test per vehicle per year of age; retests excluded",
            "age": "Years from first use (registration date if missing), rounded to nearest year",
            "generations": "Assigned by year of first use; boundaries approximate",
            "min_tests_per_point": MIN_TESTS,
            "interval": "95% Wilson score interval",
            "total_tests": total_tests,
        },
        "series": [
            {"generation": g, "points": series[g]}
            for g in GEN_ORDER if g in series
        ],
    }

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(json.dumps(output, indent=2))
    print(f"Wrote {OUT_PATH} ({total_tests:,} tests, {len(output['series'])} generations)")


if __name__ == "__main__":
    main()
