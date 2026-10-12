import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from api import app  # noqa: E402


def test_health():
    client = app.test_client()
    r = client.get("/health")
    assert r.status_code == 200
    assert r.get_json()["status"] == "ok"


def test_predict_setosa():
    client = app.test_client()
    r = client.post("/predict", json={"features": [5.1, 3.5, 1.4, 0.2]})
    assert r.status_code == 200
    assert r.get_json()["prediction"] == 0


def test_predict_virginica():
    client = app.test_client()
    r = client.post("/predict", json={"features": [6.7, 3.0, 5.2, 2.3]})
    assert r.status_code == 200
    assert r.get_json()["prediction"] == 2


def test_predict_bad_input():
    client = app.test_client()
    r = client.post("/predict", json={"features": [1, 2]})
    assert r.status_code == 400