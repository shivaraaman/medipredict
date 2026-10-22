---
title: MediPredict
emoji: 🩺
colorFrom: green
colorTo: blue
sdk: docker
app_port: 7860
pinned: false
---

# MediPredict — AI-based Multi-Disease Risk Prediction

Final-year project in Healthcare Informatics. A Flask web application that estimates the risk of
**diabetes**, **heart disease** and **chronic kidney disease** from routine clinical measurements,
using an Artificial Neural Network (Multilayer Perceptron) benchmarked against six machine-learning
algorithms, with per-patient explanations.

| Disease | Dataset | Deployed model | Test accuracy | Sensitivity | ROC-AUC |
|---|---|---|---|---|---|
| Diabetes | Pima Indians Diabetes (768) | ANN (MLP 64-32) | 76.0% | 75.9% | 0.820 |
| Heart disease | UCI Cleveland (303) | Logistic Regression | 86.9% | 92.9% | 0.961 |
| Chronic kidney disease | UCI CKD (400) | ANN (MLP 32-16) | 97.5% | 96.0% | 0.999 |

All figures are on a 20% held-out test set. See `reports/metrics.json` for every model and metric.

## Project structure

```
app.py                 Flask web app (routes, SQLite storage, JSON API, health check)
ml/diseases.py         dataset loaders, feature definitions, preprocessing
ml/train.py            training + evaluation pipeline (7 models × 3 diseases)
ml/predictor.py        inference and per-patient explanation
ml/plots.py            report figures
data/                  raw datasets
models/                trained model bundles (*.joblib)
reports/               metrics.json and figures used by the website and report
templates/, static/    web front end
tests/                 automated tests (pytest)
```

## Run locally

```bash
python -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements-dev.txt
python app.py                                           # open http://localhost:5000
pytest -q                                               # run the tests
```

## Retrain the models

```bash
python -m ml.train        # ~2 minutes; rewrites models/, reports/metrics.json and reports/figures/
pytest -q                 # check the app still works with the new models
```

## Deploy (free)

### Option A — Render (recommended)
1. Push this folder to a GitHub repository.
2. On https://render.com sign in with GitHub → **New → Blueprint** → choose the repository.
   `render.yaml` configures everything (build, start command, health check, secret key).
3. Wait for the build; the site is live at `https://medipredict-xxxx.onrender.com`.

Free instances sleep after 15 minutes idle (first visit takes ~30 s) and the SQLite history
resets on each redeploy. For permanent history, use a paid instance with a disk mounted at
`/var/data` and set `DATABASE_PATH=/var/data/medipredict.db`.

### Option B — Hugging Face Spaces
1. On https://huggingface.co create a **New Space** → SDK **Docker** → Blank.
2. Push this folder to the Space's git repository (the YAML header above configures it).
   Large binary files (`models/*.joblib`, `reports/figures/*.png`) need `git lfs track "*.joblib" "*.png"`.

### Environment variables
| Variable | Purpose | Default |
|---|---|---|
| `SECRET_KEY` | Flask session signing | dev value — **set in production** |
| `DATABASE_PATH` | SQLite file for assessment history | `instance/medipredict.db` |
| `HISTORY_LIMIT` | rows shown on the History page | `100` |
| `PORT` | listening port | `5000` (Docker: `7860`) |

## Maintenance
- **Monitoring:** the host polls `GET /health` (checks models load and the database answers). Logs record every assessment id, disease and probability — no personal data.
- **Backups:** copy the SQLite file at `DATABASE_PATH` (`sqlite3 medipredict.db ".backup backup.db"`).
- **Updating models:** retrain, run tests, commit `models/` and `reports/`, push — Render/HF redeploy automatically. CI (`.github/workflows/ci.yml`) runs the tests on every push.
- **Dependencies:** `scikit-learn` is pinned to the exact version used for training; if you upgrade it, retrain.
- **Privacy:** the app stores only an optional free-text patient reference (never names) and the entered measurements.

## API

```bash
curl -X POST https://<your-site>/api/predict/heart -H "Content-Type: application/json" \
  -d '{"age":63,"sex":1,"cp":4,"trestbps":145,"chol":233,"fbs":1,"restecg":2,"thalach":150,"exang":0,"oldpeak":2.3,"slope":3}'
```

## Disclaimer
For education and screening support only. Not a medical device; not a diagnosis.

## Data sources
- Smith et al. (1988), Pima Indians Diabetes Database — National Institute of Diabetes and Digestive and Kidney Diseases.
- Detrano et al. (1989), UCI Heart Disease (Cleveland) — Cleveland Clinic Foundation.
- Rubini, Soundarapandian & Eswaran (2015), UCI Chronic Kidney Disease — Apollo Hospitals, Karaikudi.
