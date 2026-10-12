import joblib
from sklearn.datasets import load_iris

model = joblib.load("models/iris_model.pkl")
iris = load_iris()

sample = iris.data[0]
p = model.predict([sample])

print("Features:", sample)
print("Prediction:", p[0], iris.target_names[p[0]])