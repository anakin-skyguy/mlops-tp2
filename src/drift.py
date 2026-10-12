import sys

import pandas as pd
from sklearn.datasets import load_iris

# Threshold in training std devs. Default 1.0, override: python src/drift.py 2.0
THRESHOLD = float(sys.argv[1]) if len(sys.argv) > 1 else 1.0

iris = load_iris()
train = pd.DataFrame(iris.data, columns=iris.feature_names)

# Simulated "new production data": petal length +50%, sepal width -20%
new = train.copy()
new["petal length (cm)"] = new["petal length (cm)"] * 1.5
new["sepal width (cm)"] = new["sepal width (cm)"] * 0.8

summary = pd.DataFrame({
    "train_mean": train.mean(),
    "new_mean": new.mean(),
    "train_std": train.std(),
    "new_std": new.std(),
    "train_max": train.max(),
    "new_max": new.max(),
})
summary["shift_in_std"] = (
    (summary["new_mean"] - summary["train_mean"]).abs() / summary["train_std"]
)
out_of_range = (new < train.min()) | (new > train.max())
summary["pct_out_of_range"] = out_of_range.mean() * 100

pd.set_option("display.width", 250)
pd.set_option("display.max_columns", None)

print("Threshold used:", THRESHOLD, "std devs")
print()
print("=== Training data ===")
print(train.describe().round(2))
print()
print("=== New data ===")
print(new.describe().round(2))
print()
print("=== Drift summary ===")
print(summary.round(2))
print()

drifted = summary[summary["shift_in_std"] > THRESHOLD]
if drifted.empty:
    print("OK: no feature drifted beyond the threshold.")
else:
    print("ALERT: drift detected on:", ", ".join(drifted.index))
    print("Action: check the data source, evaluate the model on recent labeled data, retrain if accuracy drops.")