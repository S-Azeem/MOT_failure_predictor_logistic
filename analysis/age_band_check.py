"""
Confounding check: fail rate by age band, split by fuel type and by the
number of advisories at the previous test.

This is the check that found Simpson's paradox: petrol 3 Series fail more
overall, but diesels fail as often or more within every age band.

Run from the repository root:
    python -m analysis.age_band_check
"""
import pandas as pd

from config import TRAINING_CSV

df = pd.read_csv(TRAINING_CSV)
df["age_band"] = pd.cut(df["age_years"], [0, 5, 8, 11, 15, 40])
df["adv_band"] = pd.cut(df["prev_advisories"], [-1, 0, 2, 4, 100],
                        labels=["0", "1-2", "3-4", "5+"])

main_fuels = df[df["fuel_type"].isin(["DIESEL", "PETROL"])]
print("Fail % by age and fuel:")
print((main_fuels.pivot_table(index="age_band", columns="fuel_type",
                              values="target_failed", aggfunc="mean",
                              observed=False) * 100).round(1), "\n")

print("Fail % by age and previous advisories:")
print((df.pivot_table(index="age_band", columns="adv_band",
                      values="target_failed", aggfunc="mean",
                      observed=False) * 100).round(1))
