"""
Failure model, step 1: build the training table (BMW 3 Series pilot).

One row per MOT test cycle. Target = did this cycle's first test fail?
Features come ONLY from earlier cycles (no leakage), plus static vehicle facts.

Advisory categories from the previous test:
  - advisories on tests from the rule change (May 2018) use their MOT manual
    section code, e.g. (5.2.3 (e))
  - earlier advisories use keyword matching: they carry codes too, but from the
    old manual, which numbered sections differently (brakes were 3, tyres 4)
  - tester notes (child seat fitted, undertrays, COVID extension...) are excluded
It also prints how often the keyword rules agree with the codes.

Phase 1 features (all from earlier cycles only):
  mileage:        prev_odometer, annual_miles_last_interval, projected_miles
  track record:   fails_last_3, n_prior_fails_pre2018, n_prior_fails_post2018,
                  prev_test_pre2018
  advisories:     advisory_trend, n_recurring_categories
The three features that used the odometer reading AT the predicted test
(odometer_miles, miles_since_prev, miles_per_year) are no longer built.

A fully commented teaching copy is in docs/build_training_annotated.py.

Run from the repository root:
    python -m pipeline.failure_model.build_training
Output: data/processed/training_3series.csv
"""
import csv
from collections import defaultdict

from config import RETEST_GAP_DAYS, RULES_CHANGE, SNAPSHOT_END, TRAINING_CSV
from pipeline.common import connect_db

TARGET_START = RULES_CHANGE   # predict only tests under the post-2018 rules
TARGET_END = SNAPSHOT_END     # bulk snapshot end
CODES_FROM = RULES_CHANGE     # new-manual section codes only from this date
OUT_PATH = TRAINING_CSV

CATEGORIES = ["tyres", "brakes", "suspension", "steering", "leaks", "other"]
FLAGS = ["corrosion", "tyre_limit"]   # cut across categories

# ---- Advisory classification rules (SQL expressions on d.text) ---------------
# LIKE is case-insensitive in SQLite for plain letters, which suits free text.

NOTE = """(d.text LIKE '%child seat%' OR d.text LIKE '%undertray%'
        OR d.text LIKE '%under tray%' OR d.text LIKE '%items removed%'
        OR d.text LIKE '%covid%' OR d.text LIKE '%headlight adjuster%'
        OR d.text LIKE '%headlamp adjuster%')"""

CODED = "(d.text GLOB '*([0-9].[0-9]*')"

CODE_CASE = """(CASE
    WHEN d.text LIKE '%(5.2.%' THEN 'tyres'
    WHEN d.text LIKE '%(5.1.%' OR d.text LIKE '%(5.3.%' THEN 'suspension'
    WHEN d.text LIKE '%(1.%' THEN 'brakes'
    WHEN d.text LIKE '%(2.%' THEN 'steering'
    WHEN d.text LIKE '%(8.4.%' THEN 'leaks'
    ELSE 'other' END)"""

KEYWORD_CASE = """(CASE
    WHEN d.text LIKE '%tyre%' THEN 'tyres'
    WHEN d.text LIKE '%brake%' THEN 'brakes'
    WHEN d.text LIKE '%steering%' OR d.text LIKE '%track rod%' THEN 'steering'
    WHEN d.text LIKE '%exhaust%' THEN 'other'
    WHEN d.text LIKE '%leak%' THEN 'leaks'
    WHEN d.text LIKE '%shock absorber%' OR d.text LIKE '%suspension%'
      OR d.text LIKE '%spring%' OR d.text LIKE '%anti-roll%'
      OR d.text LIKE '%sub-frame%' OR d.text LIKE '%subframe%'
      OR d.text LIKE '%bush%' OR d.text LIKE '%ball joint%'
      OR d.text LIKE '%wheel bearing%' THEN 'suspension'
    ELSE 'other' END)"""

# ---- Shared starting point: 3 Series vehicles and their GB tests -------------
# CROSS JOIN forces SQLite to start from the small table and use the indexes.
BASE_CTES = """
WITH veh AS (
  SELECT registration,
         substr(COALESCE(first_used_date, registration_date),1,10) AS start_date,
         UPPER(TRIM(fuel_type)) AS fuel_type,
         engine_size
  FROM vehicles
  WHERE make = 'BMW'
    AND (model = '3 SERIES' OR model GLOB '3[0-9][0-9]*')
    AND COALESCE(first_used_date, registration_date) IS NOT NULL
),
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
)"""

cat_sums = ",\n         ".join(f"SUM(cat = '{c}') AS adv_{c}" for c in CATEGORIES)
cat_coalesce = ",\n         ".join(f"COALESCE(i.adv_{c},0) AS adv_{c}" for c in CATEGORIES + FLAGS)
cat_lags = ",\n         ".join(f"LAG(c.adv_{c}) OVER w AS prev_adv_{c}" for c in CATEGORIES + FLAGS)

# A category "recurs" if it appeared on BOTH of the two previous tests.
RECURRING = ["tyres", "brakes", "suspension", "steering", "leaks", "corrosion"]
recurring_sum = "\n           + ".join(
    f"(LAG(c.adv_{c}, 1) OVER w > 0 AND LAG(c.adv_{c}, 2) OVER w > 0)" for c in RECURRING)

QUERY = BASE_CTES + f""",
classified AS (
  SELECT d.mot_test_number, d.type,
         CASE WHEN d.type = 'ADVISORY' THEN
           CASE WHEN {NOTE} THEN 'note'
                WHEN {CODED} AND t.d >= :codes_from THEN {CODE_CASE}
                ELSE {KEYWORD_CASE} END
         END AS cat,
         (d.type = 'ADVISORY' AND d.text LIKE '%corro%') AS is_corrosion,
         (d.type = 'ADVISORY' AND d.text LIKE '%legal limit%') AS is_tyre_limit
  FROM tests t
  CROSS JOIN defects d
  WHERE d.mot_test_number = t.mot_test_number
),
items AS (
  SELECT mot_test_number,
         SUM(type = 'ADVISORY') AS n_advisory,
         SUM(type IN ('FAIL','MAJOR','DANGEROUS','PRS')) AS n_fail_items,
         SUM(type = 'MINOR') AS n_minor,
         {cat_sums},
         SUM(is_corrosion) AS adv_corrosion,
         SUM(is_tyre_limit) AS adv_tyre_limit
  FROM classified
  GROUP BY mot_test_number
),
ordered AS (
  SELECT t.*,
         COALESCE(i.n_advisory,0) AS n_advisory,
         COALESCE(i.n_fail_items,0) AS n_fail_items,
         COALESCE(i.n_minor,0) AS n_minor,
         {cat_coalesce},
         LAG(t.d) OVER (PARTITION BY t.registration
                        ORDER BY t.completed_date, t.failed DESC) AS prev_any_d
  FROM tests t
  LEFT JOIN items i ON i.mot_test_number = t.mot_test_number
),
cycles AS (          -- drop retests: keep only the first test of each cycle
  SELECT * FROM ordered
  WHERE prev_any_d IS NULL
     OR julianday(d) - julianday(prev_any_d) > :gap
),
feat AS (
  SELECT c.registration, c.mot_test_number, c.d AS test_date,
         c.failed AS target_failed,
         v.fuel_type, v.engine_size,
         CAST(substr(v.start_date,1,4) AS INT) AS reg_year,
         ROUND((julianday(c.d) - julianday(v.start_date)) / 365.25, 2) AS age_years,
         ROW_NUMBER() OVER w - 1 AS n_prior_cycles,
         SUM(c.failed) OVER (w ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING) AS n_prior_fails,
         LAG(c.failed)       OVER w AS prev_failed,
         LAG(c.n_advisory)   OVER w AS prev_advisories,
         LAG(c.n_fail_items) OVER w AS prev_fail_items,
         LAG(c.n_minor)      OVER w AS prev_minors,
         {cat_lags},
         julianday(c.d) - julianday(LAG(c.d) OVER w) AS days_since_prev,
         -- Phase 1: mileage, from readings at EARLIER tests only
         LAG(c.miles, 1) OVER w AS prev_odometer,
         LAG(c.miles, 1) OVER w - LAG(c.miles, 2) OVER w AS _miles_last_interval,
         julianday(LAG(c.d, 1) OVER w) - julianday(LAG(c.d, 2) OVER w) AS _days_last_interval,
         -- Phase 1: track record
         SUM(c.failed) OVER (w ROWS BETWEEN 3 PRECEDING AND 1 PRECEDING) AS fails_last_3,
         SUM(CASE WHEN c.d < :rules_change THEN c.failed ELSE 0 END)
             OVER (w ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING) AS n_prior_fails_pre2018,
         SUM(CASE WHEN c.d >= :rules_change THEN c.failed ELSE 0 END)
             OVER (w ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING) AS n_prior_fails_post2018,
         (LAG(c.d) OVER w < :rules_change) AS prev_test_pre2018,
         -- Phase 1: advisory patterns (need two earlier cycles, else blank)
         LAG(c.n_advisory, 1) OVER w - LAG(c.n_advisory, 2) OVER w AS advisory_trend,
         CASE WHEN LAG(c.d, 2) OVER w IS NULL THEN NULL ELSE
           {recurring_sum}
         END AS n_recurring_categories
  FROM cycles c
  JOIN veh v ON v.registration = c.registration
  WINDOW w AS (PARTITION BY c.registration ORDER BY c.completed_date)
)
SELECT * FROM feat
WHERE test_date >= :target_start
ORDER BY registration, test_date
"""

# Do the keyword rules agree with the official codes, where both exist?
AGREEMENT_QUERY = BASE_CTES + f"""
SELECT {CODE_CASE} AS code_cat, {KEYWORD_CASE} AS keyword_cat, COUNT(*) AS n
FROM tests t
CROSS JOIN defects d
WHERE d.mot_test_number = t.mot_test_number
  AND d.type = 'ADVISORY'
  AND {CODED}
  AND t.d >= :codes_from
  AND NOT {NOTE}
GROUP BY 1, 2
"""


def check_keyword_agreement(con) -> None:
    rows = con.execute(AGREEMENT_QUERY, {"target_end": TARGET_END,
                                         "codes_from": CODES_FROM}).fetchall()
    total = sum(n for _, _, n in rows)
    agree = sum(n for code, kw, n in rows if code == kw)
    print(f"Keyword rules vs post-2018 codes (coded advisories, n={total:,}): "
          f"{100*agree/total:.1f}% agree")
    by_code = defaultdict(lambda: [0, 0])
    for code, kw, n in rows:
        by_code[code][0] += n
        by_code[code][1] += n if code == kw else 0
    for code in CATEGORIES:
        if code in by_code:
            tot, ok = by_code[code]
            print(f"  {code:>11}: {100*ok/tot:5.1f}% agree  (n={tot:,})")
    print("  Biggest disagreements (code -> keyword):")
    for code, kw, n in sorted((r for r in rows if r[0] != r[1]), key=lambda r: -r[2])[:5]:
        print(f"    {code} -> {kw}: {n:,}")
    print()


def main() -> None:
    con = connect_db()

    check_keyword_agreement(con)

    cur = con.execute(QUERY, {
        "target_start": TARGET_START, "target_end": TARGET_END, "gap": RETEST_GAP_DAYS,
        "codes_from": CODES_FROM, "rules_change": RULES_CHANGE,
    })
    cols = [c[0] for c in cur.description]
    rows = cur.fetchall()
    con.close()

    # Clean-up rules applied in Python so they're easy to read and change
    i = {c: k for k, c in enumerate(cols)}
    helpers = ["_miles_last_interval", "_days_last_interval"]   # used, then dropped
    keep = [c for c in cols if c not in helpers]
    out_cols = keep + ["annual_miles_last_interval", "projected_miles"]
    cleaned = []
    for r in rows:
        miles, days = r[i["_miles_last_interval"]], r[i["_days_last_interval"]]
        # Annual mileage over the last complete interval between two earlier tests.
        # Blank if either reading is missing, or the odometer went backwards.
        annual = None
        if miles is not None and days and miles >= 0:
            annual = round(miles / days * 365.25)
        # Projected mileage at this test: last reading plus usual rate x time since.
        prev_odo, gap = r[i["prev_odometer"]], r[i["days_since_prev"]]
        projected = None
        if prev_odo is not None and annual is not None and gap is not None:
            projected = round(prev_odo + annual * gap / 365.25)
        cleaned.append([r[i[c]] for c in keep] + [annual, projected])
    cols = out_cols
    i = {c: k for k, c in enumerate(cols)}

    assert cleaned, "No rows - check the database and filters"
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with OUT_PATH.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(cols)
        w.writerows(cleaned)

    # ---- Quick signal checks on the new features ------------------------------
    n = len(cleaned)
    fails = sum(r[i["target_failed"]] for r in cleaned)
    print(f"Wrote {OUT_PATH.name}: {n:,} test cycles, overall fail rate {100*fails/n:.1f}%\n")

    def rate_by(label, key_fn):
        groups = defaultdict(lambda: [0, 0])
        for r in cleaned:
            k = key_fn(r)
            groups[k][0] += 1
            groups[k][1] += r[i["target_failed"]]
        print(label)
        for k in sorted(groups, key=str):
            tot, fl = groups[k]
            print(f"  {str(k):>14}: {100*fl/tot:5.1f}% fail  (n={tot:,})")
        print()

    def banded(col, edges, labels):
        def key(r):
            v = r[i[col]]
            if v is None:
                return "no history"
            for edge, label in zip(edges, labels):
                if v <= edge:
                    return label
            return labels[-1]
        return key

    rate_by("Fails in the last 3 cycles:", banded("fails_last_3", [0, 1, 2], ["0", "1", "2", "3"]))
    rate_by("Advisory categories recurring on both previous tests:",
            banded("n_recurring_categories", [0, 1], ["0", "1", "2+"]))
    rate_by("Advisory trend (previous minus the one before):",
            banded("advisory_trend", [-1, 0], ["falling", "same", "rising"]))
    rate_by("Previous test under the pre-2018 rules:",
            banded("prev_test_pre2018", [0], ["no", "yes"]))
    rate_by("Annual miles over the last interval:",
            banded("annual_miles_last_interval", [5000, 10000, 15000],
                   ["<5k", "5-10k", "10-15k", "15k+"]))

if __name__ == "__main__":
    main()
