"""
MediPredict — AI-based multi-disease risk prediction (Flask web application).

Routes
  /                         home
  /predict/<disease>        assessment form (GET) and result (POST)
  /history                  saved assessments (SQLite)
  /performance              model evaluation dashboard
  /about                    project, datasets, disclaimer
  /api/predict/<disease>    JSON API (POST)
  /health                   health check for the hosting platform
"""
import json
import logging
import math
import os
import sqlite3
import uuid
from datetime import datetime, timezone
from pathlib import Path

from flask import (Flask, abort, flash, g, jsonify, redirect, render_template, request,
                   send_from_directory, url_for)

from ml.diseases import DISEASES
from ml.predictor import load_bundle, parse_inputs, predict

BASE = Path(__file__).resolve().parent
METRICS = json.loads((BASE / "reports" / "metrics.json").read_text())

app = Flask(__name__)
app.config.update(
    SECRET_KEY=os.environ.get("SECRET_KEY", "dev-only-change-me"),
    DATABASE=os.environ.get("DATABASE_PATH", str(BASE / "instance" / "medipredict.db")),
    HISTORY_LIMIT=int(os.environ.get("HISTORY_LIMIT", "100")),
)
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("medipredict")

SCHEMA = """
CREATE TABLE IF NOT EXISTS assessments (
    id          TEXT PRIMARY KEY,
    created_at  TEXT NOT NULL,
    disease     TEXT NOT NULL,
    inputs      TEXT NOT NULL,
    probability REAL NOT NULL,
    band        TEXT NOT NULL,
    model_name  TEXT NOT NULL,
    patient_ref TEXT
);
CREATE INDEX IF NOT EXISTS idx_assess_created ON assessments(created_at DESC);
"""


# ---------------------------------------------------------------- database
def get_db():
    if "db" not in g:
        Path(app.config["DATABASE"]).parent.mkdir(parents=True, exist_ok=True)
        g.db = sqlite3.connect(app.config["DATABASE"])
        g.db.row_factory = sqlite3.Row
    return g.db


@app.teardown_appcontext
def close_db(_exc):
    db = g.pop("db", None)
    if db is not None:
        db.close()


def init_db():
    with app.app_context():
        get_db().executescript(SCHEMA)
        get_db().commit()


def save_assessment(disease, values, result, patient_ref):
    rid = uuid.uuid4().hex[:12]
    db = get_db()
    db.execute("INSERT INTO assessments VALUES (?,?,?,?,?,?,?,?)",
               (rid, datetime.now(timezone.utc).isoformat(timespec="seconds"), disease,
                json.dumps(values), result["probability"], result["band"], result["model_name"],
                patient_ref or None))
    db.commit()
    return rid


# ---------------------------------------------------------------- template helpers
@app.context_processor
def inject_globals():
    return {"DISEASES": DISEASES, "year": datetime.now().year}


@app.template_filter("pct")
def pct(v, digits=1):
    return f"{v * 100:.{digits}f}%"


@app.template_filter("fmt")
def fmt(v):
    if v is None:
        return "—"
    return f"{float(v):.4g}" if isinstance(v, (int, float)) else str(v)


def display_value(feature, v):
    if v is None:
        return "Not provided"
    if feature["kind"] == "select":
        for val, lab in feature["options"]:
            if float(val) == float(v):
                return lab
    unit = f" {feature['unit']}" if feature.get("unit") else ""
    return f"{v:g}{unit}"


app.jinja_env.globals["display_value"] = display_value
app.jinja_env.filters["cos"] = math.cos
app.jinja_env.filters["sin"] = math.sin


def check_disease(disease):
    if disease not in DISEASES:
        abort(404)


# ---------------------------------------------------------------- pages
@app.route("/")
def index():
    summary = {k: {"model": m["deployed_model"], **m["results"][m["deployed_model"]]} for k, m in METRICS.items()}
    return render_template("index.html", summary=summary, metrics=METRICS)


@app.route("/predict/<disease>", methods=["GET", "POST"])
def assess(disease):
    check_disease(disease)
    spec = DISEASES[disease]
    if request.method == "GET":
        sample = {f["key"]: f["example"] for f in spec["features"]} if request.args.get("sample") else {}
        return render_template("form.html", disease=disease, spec=spec, values=sample, errors={})

    values, errors = parse_inputs(disease, request.form)
    if errors:
        flash("Please correct the highlighted fields.", "error")
        return render_template("form.html", disease=disease, spec=spec, values=request.form, errors=errors), 400
    result = predict(disease, values)
    rid = save_assessment(disease, values, result, request.form.get("patient_ref", "").strip()[:40])
    log.info("assessment %s disease=%s prob=%.3f", rid, disease, result["probability"])
    return redirect(url_for("result", rid=rid))


@app.route("/result/<rid>")
def result(rid):
    row = get_db().execute("SELECT * FROM assessments WHERE id=?", (rid,)).fetchone()
    if row is None:
        abort(404)
    disease = row["disease"]
    values = json.loads(row["inputs"])
    res = predict(disease, values)
    return render_template("result.html", disease=disease, spec=DISEASES[disease], values=values,
                           res=res, row=row, metrics=METRICS[disease])


@app.route("/history")
def history():
    rows = get_db().execute("SELECT * FROM assessments ORDER BY created_at DESC LIMIT ?",
                            (app.config["HISTORY_LIMIT"],)).fetchall()
    return render_template("history.html", rows=rows)


@app.post("/history/<rid>/delete")
def delete_assessment(rid):
    db = get_db()
    db.execute("DELETE FROM assessments WHERE id=?", (rid,))
    db.commit()
    flash("Assessment deleted.", "ok")
    return redirect(url_for("history"))


@app.route("/performance")
def performance():
    return render_template("performance.html", metrics=METRICS)


@app.route("/about")
def about():
    return render_template("about.html", metrics=METRICS)


@app.route("/figures/<path:name>")
def figures(name):
    return send_from_directory(BASE / "reports" / "figures", name)


# ---------------------------------------------------------------- API
@app.post("/api/predict/<disease>")
def api_predict(disease):
    if disease not in DISEASES:
        return jsonify(error=f"Unknown disease '{disease}'. Use one of {list(DISEASES)}"), 404
    payload = request.get_json(silent=True) or {}
    values, errors = parse_inputs(disease, payload)
    if errors:
        return jsonify(error="validation_failed", fields=errors), 400
    res = predict(disease, values)
    return jsonify(disease=disease, probability=round(res["probability"], 4), threshold=res["threshold"],
                   prediction=DISEASES[disease]["positive" if res["positive"] else "negative"],
                   risk_band=res["band"], model=res["model_name"],
                   top_factors=[{"feature": c["feature"], "effect": round(c["delta"], 4)}
                                for c in res["contributions"]])


@app.route("/health")
def health():
    try:
        for d in DISEASES:
            load_bundle(d)
        get_db().execute("SELECT 1")
        return jsonify(status="ok", models=list(DISEASES))
    except Exception as e:  # pragma: no cover
        log.exception("health check failed")
        return jsonify(status="error", detail=str(e)), 500


@app.errorhandler(404)
def not_found(_e):
    return render_template("error.html", code=404, message="That page doesn't exist."), 404


@app.errorhandler(500)
def server_error(_e):
    return render_template("error.html", code=500, message="Something went wrong on our side."), 500


init_db()
for _d in DISEASES:  # warm the model cache at start-up
    load_bundle(_d)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)), debug=bool(os.environ.get("DEBUG")))
