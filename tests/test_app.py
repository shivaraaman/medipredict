import os
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
os.environ["DATABASE_PATH"] = os.path.join(tempfile.mkdtemp(), "test.db")

import pytest  # noqa: E402

from app import app  # noqa: E402
from ml.diseases import DISEASES  # noqa: E402


@pytest.fixture
def client():
    app.config["TESTING"] = True
    with app.test_client() as c:
        yield c


def sample(d):
    return {f["key"]: f["example"] for f in DISEASES[d]["features"]}


@pytest.mark.parametrize("path", ["/", "/performance", "/about", "/history", "/health",
                                  "/predict/diabetes", "/predict/heart?sample=1", "/predict/kidney",
                                  "/figures/heart_roc.png"])
def test_pages(client, path):
    assert client.get(path).status_code == 200


def test_404(client):
    assert client.get("/predict/cancer").status_code == 404


@pytest.mark.parametrize("d", list(DISEASES))
def test_form_submit_and_history(client, d):
    r = client.post(f"/predict/{d}", data={**sample(d), "patient_ref": "T-1"})
    assert r.status_code == 302
    page = client.get(r.headers["Location"])
    assert page.status_code == 200 and b"risk result" in page.data
    assert b"T-1" in client.get("/history").data


def test_validation(client):
    r = client.post("/predict/diabetes", data={"Glucose": "abc"})
    assert r.status_code == 400 and b"required" in r.data


def test_api_high_vs_low(client):
    low = client.post("/api/predict/diabetes", json={"Pregnancies": 0, "Glucose": 85, "BloodPressure": 66,
                                                     "BMI": 22, "DiabetesPedigreeFunction": 0.2, "Age": 24}).json
    high = client.post("/api/predict/diabetes", json={"Pregnancies": 8, "Glucose": 196, "BloodPressure": 76,
                                                      "BMI": 39, "DiabetesPedigreeFunction": 1.1, "Age": 55}).json
    assert low["probability"] < 0.3 < 0.7 < high["probability"]
    assert high["risk_band"] == "high" and low["risk_band"] == "low"


def test_kidney_optional_blank(client):
    vals = {k: v for k, v in sample("kidney").items()
            if any(f["key"] == k and f["required"] for f in DISEASES["kidney"]["features"])}
    r = client.post("/api/predict/kidney", json=vals)
    assert r.status_code == 200 and 0 <= r.json["probability"] <= 1
