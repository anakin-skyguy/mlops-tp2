import os
import sys

import joblib
import mlflow
import mlflow.sklearn
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

# Number of trees: pass it as an argument, e.g. python src/train.py 50
n_estimators = int(sys.argv[1]) if len(sys.argv) > 1 else 100

mlflow.set_experiment("Iris_MLOps")

iris = load_iris()
X, y = iris.data, iris.target
Xtr, Xte, ytr, yte = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

with mlflow.start_run():
    model = RandomForestClassifier(n_estimators=n_estimators, random_state=42)
    model.fit(Xtr, ytr)
    pred = model.predict(Xte)
    acc = accuracy_score(yte, pred)

    mlflow.log_param("n_estimators", n_estimators)
    mlflow.log_metric("accuracy", acc)
    mlflow.sklearn.log_model(
        model,
        name="model",
        skops_trusted_types=["sklearn.tree._tree.Tree"],
    )

    os.makedirs("models", exist_ok=True)
    joblib.dump(model, "models/iris_model.pkl")

    print("n_estimators:", n_estimators)
    print("Accuracy:", acc)