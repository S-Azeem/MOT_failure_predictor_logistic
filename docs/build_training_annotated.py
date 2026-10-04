"""
=====================================================================
 build_training_annotated.py  -  A READING COPY, NOT PART OF THE PIPELINE
 The feature-engineering script, with teaching comments.
 Snapshot of pipeline/failure_model/build_training.py (October 2026).
 The pipeline version is the one to run and to edit; this copy is for
 understanding how it works and may fall behind it.
=====================================================================

THE GOAL
--------
Turn three raw tables (vehicles, mot_tests, defects) into ONE table where:
  - each row  = one car's MOT "test cycle" (its first test in a given year)
  - the target = did that test fail? (1 = fail, 0 = pass)
  - the features = only things known BEFORE that test (no peeking at the answer)

A WORKED EXAMPLE TO KEEP IN MIND
--------------------------------
Imagine one car, a 2012 BMW 320D, with this raw history:

  Test  Date         Result   Advisories found at that test
  ----  -----------  -------  ------------------------------------------
  T1    2016-03-01   PASS     (none)
  T2    2017-03-01   PASS     "tyre worn close to legal limit", "brake pipe corroded"
  T3    2019-03-01   FAIL     ...
  T4    2019-03-05   PASS     (the retest, 4 days after the fail)
  T5    2020-03-01   PASS     ...

The script turns this into training rows like:

  Row for T3 (2019):  target = 1 (failed)
                      prev_failed = 0          <- T2 passed
                      prev_advisories = 2      <- T2 had two advisories
                      prev_adv_tyres = 1, prev_adv_brakes = 1, prev_adv_corrosion = 1
                      n_prior_fails = 0        <- no fails before T3
                      age_years = 7.0

  Row for T5 (2020):  target = 0 (passed)
                      prev_failed = 1          <- T3 failed (T4, the retest, is ignored)
                      n_prior_fails = 1

  T1 and T2 get no rows of their own because they're before May 2018,
  but they still feed the history features of later rows.
  T4 gets no row at all: it's a retest, not a new cycle.

Keep this car in mind while reading each step below.

HOW THE SCRIPT IS ORGANISED
---------------------------
  Part 1  Settings
  Part 2  Rules for sorting advisory text into categories
  Part 3  The big SQL query, built in 8 steps (CTEs), each feeding the next
  Part 4  A check that the keyword rules agree with the official codes
  Part 5  Python: run the query, tidy up, save the CSV, print quick checks

A CTE ("common table expression") is a named, temporary result inside one
query: WITH step_a AS (...), step_b AS (... uses step_a ...). Think of each
CTE as one stage of a pipeline, like a variable holding an intermediate table.

Run:  python build_training_annotated.py
"""
import csv                          # writing the output file
import sqlite3                      # talking to the SQLite database
from collections import defaultdict # a dict that creates missing keys automatically
from pathlib import Path            # tidy, cross-platform file paths


# =====================================================================
# PART 1: SETTINGS
# Everything you might want to change lives here, not buried in the code.
# =====================================================================

HERE = Path(__file__).parent                 # the folder this script is in
DB_PATH = HERE / "mot_bulk_1pct.db"          # the database (read-only)
OUT_PATH = HERE / "training_3series.csv"     # where the training table goes

TARGET_START = "2018-05-20"   # we only PREDICT tests after the May 2018 rule change,
                              # because "fail" meant something different before it
TARGET_END = "2026-02-04"     # the bulk snapshot ends here; later tests only exist
                              # for API-refreshed cars, a skewed subset
RETEST_GAP_DAYS = 60          # a test within 60 days of the previous one = a retest
CODES_FROM = "2018-05-20"     # only trust the manual section codes from this date;
                              # older advisories use the OLD manual's numbering

CATEGORIES = ["tyres", "brakes", "suspension", "steering", "leaks", "other"]
FLAGS = ["corrosion", "tyre_limit"]   # these cut across categories
                                      # (a corroded brake pipe is "brakes" AND corrosion)


# =====================================================================
# PART 2: RULES FOR SORTING ADVISORY TEXT
# These are small pieces of SQL, stored as Python strings, that get pasted
# into the big query in Part 3. Keeping them separate makes them easy to
# read and to reuse in the agreement check (Part 4).
#
# d.text = the advisory's free text, e.g.
#   "Nearside Rear Tyre worn close to legal limit/worn on edge (5.2.3 (e))"
# LIKE '%word%' = "the text contains 'word'" (case-insensitive in SQLite)
# =====================================================================

# Tester notes: comments about the inspection, not about the car's condition.
# These are excluded so they don't count as "something wrong with the car".
NOTE = """(d.text LIKE '%child seat%' OR d.text LIKE '%undertray%'
        OR d.text LIKE '%under tray%' OR d.text LIKE '%items removed%'
        OR d.text LIKE '%covid%' OR d.text LIKE '%headlight adjuster%'
        OR d.text LIKE '%headlamp adjuster%')"""

# Does the text contain a section code like "(5.2.3"?
# GLOB pattern: * = anything, [0-9] = one digit.
# So '*([0-9].[0-9]*' = "somewhere there's a bracket, digit, dot, digit".
CODED = "(d.text GLOB '*([0-9].[0-9]*')"

# Rule A - use the official code (only trusted for tests from May 2018).
# The first numbers of the code say which part of the MOT manual it's from:
#   (5.2.x = tyres, 5.1/5.3 = axles and suspension, 1.x = brakes,
#    2.x = steering, 8.4 = fluid leaks). Anything else = "other".
# CASE ... WHEN ... THEN ... END is SQL's version of if / elif / else.
CODE_CASE = """(CASE
    WHEN d.text LIKE '%(5.2.%' THEN 'tyres'
    WHEN d.text LIKE '%(5.1.%' OR d.text LIKE '%(5.3.%' THEN 'suspension'
    WHEN d.text LIKE '%(1.%' THEN 'brakes'
    WHEN d.text LIKE '%(2.%' THEN 'steering'
    WHEN d.text LIKE '%(8.4.%' THEN 'leaks'
    ELSE 'other' END)"""

# Rule B - keywords (used for pre-2018 advisories and hand-typed ones).
# ORDER MATTERS: the first matching line wins. "exhaust" is checked before
# "leak" so that "exhaust leak" counts as other, not as a fluid leak.
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


# =====================================================================
# PART 3: THE BIG QUERY, IN 8 STEPS
# =====================================================================

# ---- Steps 1 and 2 are shared with the agreement check, so they're stored once.
BASE_CTES = """
-- STEP 1: veh
-- Which cars? All BMW 3 Series, under any of their 291 labels.
-- Also work out each car's "start date" for calculating age:
-- first-use date if we have it, otherwise registration date (COALESCE
-- returns the first value that isn't blank). We validated that the two
-- agree within a month for 97% of cars that have both.
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

-- STEP 2: tests
-- All of those cars' MOT tests, one row per test, with:
--   d      = the test date as plain YYYY-MM-DD text
--   failed = 1 if the test failed, else 0 (a comparison in SQLite gives 1 or 0)
--   miles  = the odometer in miles, but ONLY if the tester actually read it
--            ('READ'); km readings are converted (x 0.621371)
-- Filters: GB tests only (not Northern Ireland), only pass/fail results,
-- nothing after the snapshot end.
-- "FROM veh CROSS JOIN mot_tests WHERE ..." is the same as an inner join;
-- in SQLite, CROSS JOIN also tells the planner to start from the small veh
-- table and look each car's tests up through the index. Much faster.
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
# (":target_end" is a PARAMETER: a placeholder filled in safely when the
#  query runs, from the settings in Part 1.)

# ---- These three lines generate repetitive SQL, so we don't type it 8 times.
# For example, cat_sums becomes:
#   SUM(cat = 'tyres') AS adv_tyres,
#   SUM(cat = 'brakes') AS adv_brakes, ... and so on
cat_sums = ",\n         ".join(f"SUM(cat = '{c}') AS adv_{c}" for c in CATEGORIES)
cat_coalesce = ",\n         ".join(f"COALESCE(i.adv_{c},0) AS adv_{c}" for c in CATEGORIES + FLAGS)
cat_lags = ",\n         ".join(f"LAG(c.adv_{c}) OVER w AS prev_adv_{c}" for c in CATEGORIES + FLAGS)

QUERY = BASE_CTES + f""",

-- STEP 3: classified
-- One row per DEFECT (advisory, minor, major...) on those tests.
-- Each advisory gets a category ("cat"):
--   tester note                 -> 'note' (ignored later)
--   coded AND test is post-2018 -> use the official code (Rule A)
--   otherwise                   -> use keywords (Rule B)
-- Non-advisory defects get cat = NULL (blank).
-- Two yes/no flags are also set: corrosion, and tyre at legal limit.
-- Worked example: T2's "brake pipe corroded" is pre-2018, so keywords
-- apply -> cat = 'brakes', is_corrosion = 1.
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

-- STEP 4: items
-- Squash many defect rows into ONE row per test, by counting.
-- SUM(condition) counts the rows where the condition is true (1s add up).
-- Worked example: T2 -> n_advisory = 2, adv_tyres = 1, adv_brakes = 1,
-- adv_corrosion = 1, adv_tyre_limit = 1.
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

-- STEP 5: ordered
-- Attach those counts back onto each test. LEFT JOIN keeps tests with NO
-- defects at all (an inner join would silently drop clean passes!), and
-- COALESCE turns their blank counts into 0.
-- Also record each test's PREVIOUS test date for the same car, using a
-- WINDOW FUNCTION:
--   LAG(x) OVER (PARTITION BY car ORDER BY date) = "x from the row before,
--   within the same car's history, in date order"
-- "failed DESC" breaks ties: if a fail and a pass share a timestamp, the
-- fail counts as first.
-- Worked example: T4's previous test date is T3's (2019-03-01).
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

-- STEP 6: cycles
-- Drop retests. Keep a test only if it's the car's first ever test, or it
-- comes more than 60 days after the previous test.
-- julianday() turns a date into a number of days, so subtracting two gives
-- the gap in days.
-- Worked example: T4 is 4 days after T3 -> dropped. T1, T2, T3, T5 stay.
cycles AS (
  SELECT * FROM ordered
  WHERE prev_any_d IS NULL
     OR julianday(d) - julianday(prev_any_d) > :gap
),

-- STEP 7: feat  (THE FEATURE ENGINEERING)
-- Now each car's rows are its test cycles. For each cycle, build the target
-- and the features. The window "w" (defined at the bottom) means: this
-- car's cycles, in date order.
--
--   target_failed       = did THIS cycle fail? (the answer we predict)
--   age_years           = days since start date / 365.25
--   n_prior_cycles      = row number in the car's history, minus 1
--   n_prior_fails       = running total of fails, STOPPING at the row before
--                         ("ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING"
--                          = all earlier rows, excluding this one)
--   prev_*              = LAG = the value from the PREVIOUS cycle
--   days_since_prev     = gap since the previous cycle
--   miles_since_prev    = odometer change since the previous cycle
--
-- THIS IS WHERE LEAKAGE IS PREVENTED. Every feature looks backwards only:
-- LAG takes the previous row, and the running sum stops one row early.
-- Nothing from the cycle being predicted leaks into its own features.
--
-- Worked example: row T5 -> prev_failed = 1 (from T3, since T4 was dropped),
-- n_prior_fails = 1, days_since_prev = 366.
feat AS (
  SELECT c.registration, c.mot_test_number, c.d AS test_date,
         c.failed AS target_failed,
         v.fuel_type, v.engine_size,
         CAST(substr(v.start_date,1,4) AS INT) AS reg_year,
         ROUND((julianday(c.d) - julianday(v.start_date)) / 365.25, 2) AS age_years,
         c.miles AS odometer_miles,
         ROW_NUMBER() OVER w - 1 AS n_prior_cycles,
         SUM(c.failed) OVER (w ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING) AS n_prior_fails,
         LAG(c.failed)       OVER w AS prev_failed,
         LAG(c.n_advisory)   OVER w AS prev_advisories,
         LAG(c.n_fail_items) OVER w AS prev_fail_items,
         LAG(c.n_minor)      OVER w AS prev_minors,
         {cat_lags},
         julianday(c.d) - julianday(LAG(c.d) OVER w) AS days_since_prev,
         c.miles - LAG(c.miles) OVER w AS miles_since_prev
  FROM cycles c
  JOIN veh v ON v.registration = c.registration
  WINDOW w AS (PARTITION BY c.registration ORDER BY c.completed_date)
)

-- STEP 8: keep only the cycles we want to PREDICT (post-2018).
-- Note the order: history was built from ALL cycles first (steps 5-7),
-- and only now do we filter. Filtering earlier would throw away the
-- pre-2018 history that later rows need.
-- Worked example: T1 and T2 are dropped here, but T3 already "remembered" T2.
SELECT * FROM feat
WHERE test_date >= :target_start
ORDER BY registration, test_date
"""


# =====================================================================
# PART 4: DO THE KEYWORD RULES MATCH THE OFFICIAL CODES?
# For post-2018 advisories we have BOTH a code and keywords, so we can
# classify each one both ways and count how often they agree. High
# agreement means the keywords are trustworthy for pre-2018 advisories,
# where keywords are all we have. (This check is what caught the
# old-manual bug: agreement was 45% before the fix.)
# =====================================================================

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
    """Print how often the keyword rules agree with the official codes."""
    rows = con.execute(AGREEMENT_QUERY, {"target_end": TARGET_END,
                                         "codes_from": CODES_FROM}).fetchall()
    # each row is (category by code, category by keywords, how many advisories)
    total = sum(n for _, _, n in rows)
    agree = sum(n for code, kw, n in rows if code == kw)
    print(f"Keyword rules vs post-2018 codes (coded advisories, n={total:,}): "
          f"{100*agree/total:.1f}% agree")

    # agreement broken down by the code's category
    by_code = defaultdict(lambda: [0, 0])     # category -> [total, agreeing]
    for code, kw, n in rows:
        by_code[code][0] += n
        by_code[code][1] += n if code == kw else 0
    for code in CATEGORIES:
        if code in by_code:
            tot, ok = by_code[code]
            print(f"  {code:>11}: {100*ok/tot:5.1f}% agree  (n={tot:,})")

    # the five biggest disagreements point at which rule needs fixing
    print("  Biggest disagreements (code -> keyword):")
    for code, kw, n in sorted((r for r in rows if r[0] != r[1]), key=lambda r: -r[2])[:5]:
        print(f"    {code} -> {kw}: {n:,}")
    print()


# =====================================================================
# PART 5: RUN IT, TIDY UP, SAVE, CHECK
# =====================================================================

def main() -> None:
    # Open the database READ-ONLY, so this script can never change it.
    # (Without the check, sqlite3 would silently create an empty database
    #  if the path was wrong - which happened once!)
    if not DB_PATH.exists():
        raise SystemExit(f"Database not found: {DB_PATH.resolve()}")
    con = sqlite3.connect(DB_PATH.resolve().as_uri() + "?mode=ro", uri=True)

    # Part 4 first: if the keyword rules are badly off, you see it straight away.
    check_keyword_agreement(con)

    # Run the big query. The dict fills in the :parameters from Part 1.
    cur = con.execute(QUERY, {
        "target_start": TARGET_START, "target_end": TARGET_END, "gap": RETEST_GAP_DAYS,
        "codes_from": CODES_FROM,
    })
    cols = [c[0] for c in cur.description]   # the column names
    rows = cur.fetchall()                    # every row, as a list of tuples
    con.close()

    # --- Tidy-up rules, done in Python because they're easier to read here ---
    i = {c: k for k, c in enumerate(cols)}   # column name -> position, e.g. i["prev_failed"]
    cols.append("miles_per_year")            # one extra feature, computed below
    cleaned = []
    for r in rows:
        r = list(r)                          # tuples can't be changed; lists can
        msp, dsp = r[i["miles_since_prev"]], r[i["days_since_prev"]]

        # Odometer went BACKWARDS since last time? That's a typo, a replaced
        # clock or clocking - either way the number can't be trusted, so blank it.
        if msp is not None and msp < 0:
            r[i["miles_since_prev"]] = None
            msp = None

        # Annualise the mileage: miles per day x 365.25.
        # Blank if we don't have both numbers (e.g. a car's first test).
        r.append(round(msp / dsp * 365.25) if msp is not None and dsp else None)
        cleaned.append(r)

    # Fail loudly rather than silently writing an empty file.
    assert cleaned, "No rows - check DB_PATH and filters"

    # Save the training table.
    with OUT_PATH.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(cols)       # header row
        w.writerows(cleaned)   # data rows

    # --- Quick signal checks: does each new feature separate fails from passes? ---
    n = len(cleaned)
    fails = sum(r[i["target_failed"]] for r in cleaned)
    print(f"Wrote {OUT_PATH.name}: {n:,} test cycles, overall fail rate {100*fails/n:.1f}%\n")

    def rate_by(label, key_fn):
        """Group rows by key_fn(row) and print the fail rate in each group."""
        groups = defaultdict(lambda: [0, 0])   # group -> [cycles, fails]
        for r in cleaned:
            k = key_fn(r)
            groups[k][0] += 1
            groups[k][1] += r[i["target_failed"]]
        print(label)
        for k in sorted(groups, key=str):
            tot, fl = groups[k]
            print(f"  {str(k):>14}: {100*fl/tot:5.1f}% fail  (n={tot:,})")
        print()

    def had(col):
        """Returns a function that labels a row 'yes', 'no' or 'no history' for col."""
        def key(r):
            v = r[i[col]]
            return "no history" if v is None else ("yes" if v > 0 else "no")
        return key

    for c in CATEGORIES + FLAGS:
        rate_by(f"Previous test had a {c.replace('_', ' ')} advisory:", had(f"prev_adv_{c}"))


# Standard Python: only run main() when this file is run directly,
# not when another script imports it.
if __name__ == "__main__":
    main()
