from flask import Flask, jsonify, request
import joblib

MODEL_VERSION = "1.0"

app = Flask(__name__)
model = joblib.load("models/iris_model.pkl")


@app.get("/")
def home():
    return jsonify({"service": "Iris MLOps API", "status": "running"})


@app.get("/health")
def health():
    return jsonify({"status": "ok", "model_version": MODEL_VERSION})


@app.post("/predict")
def predict():
    data = request.get_json(silent=True)
    if not data or "features" not in data:
        return jsonify({"error": "Send JSON like {\"features\": [5.1, 3.5, 1.4, 0.2]}"}), 400

    features = data["features"]
    if not isinstance(features, list) or len(features) != 4:
        return jsonify({"error": "features must be a list of 4 numbers"}), 400

    p = model.predict([features])
    return jsonify({
        "prediction": int(p[0]),
        "model_version": MODEL_VERSION,
    })


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)