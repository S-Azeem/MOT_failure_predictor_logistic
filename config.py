"""
Project settings: the single source of truth.

Every script imports its paths and shared settings from here, so a change
(a new cut-off date, a moved database) is made once and applies everywhere.
Run scripts from the repository root as modules, e.g.
    python -m pipeline.failure_model.build_training
"""
from pathlib import Path

# ---- Paths ------------------------------------------------------------------
ROOT = Path(__file__).resolve().parent          # the repository folder

DATA_RAW = ROOT / "data" / "raw"                # source data (not in git)
DATA_PROCESSED = ROOT / "data" / "processed"    # built tables (not in git)
OUTPUTS = ROOT / "outputs"                      # charts, plots (not in git)

DB_PATH = DATA_RAW / "mot_bulk_1pct.db"         # the 1% DVSA sample
TRAINING_CSV = DATA_PROCESSED / "training_3series.csv"

SITE_DIR = Path("/Users/shaamiazeem/mot-site")  # the website repository
CHART_JSON = SITE_DIR / "data" / "failure-model" / "bmw-3-series.json"

# ---- Data rules ---------------------------------------------------------------
RULES_CHANGE = "2018-05-20"    # MOT rule change: predict only tests from here;
                               # also the date manual section codes become reliable
SNAPSHOT_END = "2026-02-04"    # bulk snapshot end; later tests are a skewed subset
RETEST_GAP_DAYS = 60           # a test within this many days of the last is a retest
MIN_TESTS_PER_POINT = 30       # smallest group shown on a chart

# ---- Train / validation / test split (by date) --------------------------------
VALIDATION_START = "2023-01-01"   # roadmap Phase 0: selection decisions use 2023
TEST_START = "2024-01-01"         # final test period; touch once at the end
