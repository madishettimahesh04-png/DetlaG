import joblib

pipe = joblib.load("pipeline.pkl")

print(pipe["selected_columns"])

print(pipe["variance_selector"])

import joblib

pipe = joblib.load("pipeline.pkl")

for col in pipe["selected_columns"]:
    if (
        "HydrogenBondClass" in col
        or "FunctionalGroup" in col
        or "Family" in col
        or "PolarityClass" in col
    ):
        print(col)