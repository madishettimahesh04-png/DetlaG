# inspect_feature_order.py

import joblib

feature_order = joblib.load("feature_order.pkl")

print("Total Features:", len(feature_order))

for i, col in enumerate(feature_order[:30]):
    print(i, col)