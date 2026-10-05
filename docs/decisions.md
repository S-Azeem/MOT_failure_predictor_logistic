# Decisions log

One line per decision: what was decided, and why. Newest at the bottom of each section.
Add reference scores and phase results as they come in.

## Data

| Date | Decision | Reason |
|---|---|---|
| 2026-09-30 | 3 Series = "3 SERIES" or any label of 3 followed by two digits; M3 excluded | 291 labels in the data; M3 is a different car |
| 2026-09-30 | Start date = first-use date, falling back to registration date | 21% lack first-use; the two agree within a month for 97.2% of cars with both |
| 2026-09-30 | Great Britain (DVSA) tests only | NI tests from age 4 at its own centres |
| 2026-09-30 | Stop all tests at 4 Feb 2026 | Bulk snapshot end; later tests exist only for API-refreshed survivors |
| 2026-09-30 | Predict only tests from 20 May 2018; earlier history still used as input | The meaning of "fail" changed with the 2018 rules |
| 2026-09-30 | Fail includes PRS | Every PRS test is already recorded as a fail; robust to dealer vs test-centre differences |
| 2026-10-01 | Retest = a test within 60 days of the previous one; dropped. A fail wins a same-timestamp tie | Retests would inflate the pass rate |
| 2026-10-02 | Advisory categories: section codes from 20 May 2018, keywords before | Pre-2018 codes come from the old manual with different numbering; keywords agree with new codes for 98.7% |
| 2026-10-05 | Known gap: 518 tyre-coded advisories don't say "tyre" | Affects pre-2018 keyword fallback only; small |

## Features

| Date | Decision | Reason |
|---|---|---|
| 2026-10-02 | Advisory categories kept in the data, not yet in the model | Added nothing to AUC in the pilot (0.658 both ways) |
| 2026-10-04 | Removed odometer_miles, miles_since_prev, miles_per_year | Use the reading taken at the predicted test, not known before it |
| 2026-10-04 | New mileage feature named annual_miles_last_interval | Clear about which interval; avoids confusion with the removed feature |

## Modelling

| Date | Decision | Reason |
|---|---|---|
| 2026-10-02 | Logistic regression is the champion | Boosting won by only 0.004 AUC in the pilot |
| 2026-10-02 | Show users a relative-risk band, with the probability beneath | Bands only need ranking, so they survive calibration drift; bands validated at 18% / 25% / 34% |
| 2026-10-05 | Three-way split: train before 2023, validation 2023, test 2024 to Feb 2026 | The pilot reused the test set for decisions; the test set now stays closed until Phase 7 |

## Reference scores (validation, Phase 0)

Copy from outputs/reference_scores.csv after running train_model.

| Model | AUC | Brier skill |
|---|---|---|
| Baseline (gen + age) | | |
| Logistic | | |
| Boosting | | |
