# Roadworthy: MOT data pipeline

Data and modelling pipeline behind [roadworthy.org.uk](https://roadworthy.org.uk), built on a
1% sample of DVSA MOT history. The website itself lives in a separate repository (`mot-site`);
this repository produces the data and models it uses.

## Layout

```
config.py               all paths and shared settings: change things here, not in scripts
pipeline/
  common.py             shared helpers: read-only database connection, 3 Series generations
  ingest/               getting data from the DVSA API and bulk files
  site_data/            builds the JSON files the website charts read
  failure_model/        the MOT failure prediction model
analysis/               one-off investigations and checks
docs/                   data dictionary, annotated teaching copy of the training build
data/                   not in git: raw/ (source databases) and processed/ (built tables)
outputs/                not in git: plots and other generated files
```

## Setup

```
source ~/Desktop/MOT/motvenv/bin/activate
pip install -r requirements.txt
```

## Running

Always run from the repository root, as modules (`python -m ...`), so every
script can import `config`.

| What | Command | Reads | Writes |
|---|---|---|---|
| Website chart data | `python -m pipeline.site_data.build_chart_data` | `data/raw/mot_bulk_1pct.db` | `mot-site/data/failure-model/bmw-3-series.json` |
| Failure model: training table | `python -m pipeline.failure_model.build_training` | `data/raw/mot_bulk_1pct.db` | `data/processed/training_3series.csv` |
| Failure model: train and evaluate | `python -m pipeline.failure_model.train_model` | `data/processed/training_3series.csv` | `outputs/calibration_3series.png` |
| Redundancy check (VIF) | `python -m analysis.redundancy_check` | `data/processed/training_3series.csv` | screen only |
| Age-band confounding check | `python -m analysis.age_band_check` | `data/processed/training_3series.csv` | screen only |

The ingest scripts in `pipeline/ingest/` were moved here as they were; check their
file paths before running them.

## Key data decisions

All set in `config.py`:

- **Rule change, 20 May 2018:** only tests from this date are predicted; history from
  earlier tests is still used as input.
- **Snapshot end, 4 Feb 2026:** later tests exist only for API-refreshed (surviving)
  cars, so they are excluded.
- **Retests:** a test within 60 days of the previous one is a retest and is dropped.
- **Population:** Great Britain (DVSA) tests only.

See `docs/training_data_dictionary.xlsx` for every column in the training table.
