"""Inference + per-patient explanation used by the web app."""
from functools import lru_cache
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from ml.diseases import DISEASES

MODEL_DIR = Path(__file__).resolve().parent.parent / "models"


@lru_cache(maxsize=None)
def load_bundle(disease):
    return joblib.load(MODEL_DIR / f"{disease}.joblib")


def parse_inputs(disease, form):
    """Validate raw form values. Returns (values dict, errors dict). Blank optional -> None."""
    values, errors = {}, {}
    for f in DISEASES[disease]["features"]:
        raw = str(form.get(f["key"], "")).strip()
        if raw == "":
            if f["required"]:
                errors[f["key"]] = "This field is required."
            values[f["key"]] = None
            continue
        try:
            v = float(raw)
        except ValueError:
            errors[f["key"]] = "Please enter a number."
            continue
        if f["kind"] == "number":
            if (f["min"] is not None and v < f["min"]) or (f["max"] is not None and v > f["max"]):
                errors[f["key"]] = f"Expected a value between {f['min']} and {f['max']}."
        else:
            allowed = {float(o[0]) for o in f["options"]}
            if v not in allowed:
                errors[f["key"]] = "Please choose one of the listed options."
        values[f["key"]] = v
    return values, errors


def risk_band(prob, thr):
    if prob >= thr:
        return "high", "High risk"
    if prob >= 0.7 * thr:
        return "borderline", "Borderline risk"
    return "low", "Low risk"


def predict(disease, values):
    b = load_bundle(disease)
    feats = b["features"]
    row = pd.DataFrame([[values.get(k) if values.get(k) is not None else np.nan for k in feats]],
                       columns=feats, dtype=float)
    prob = float(b["pipeline"].predict_proba(row)[0, 1])

    # Local explanation: how much does each entered value move the risk compared with
    # replacing it by a typical (median / most common) training value?
    labels = {f["key"]: f["label"] for f in DISEASES[disease]["features"]}
    probes = []
    entered = [k for k in feats if values.get(k) is not None]
    for k in entered:
        r = row.copy()
        r[k] = b["reference"][k]
        probes.append(r)
    contributions = []
    if probes:
        p_ref = b["pipeline"].predict_proba(pd.concat(probes, ignore_index=True))[:, 1]
        for k, pr in zip(entered, p_ref):
            delta = prob - float(pr)
            if abs(delta) >= 0.005:
                contributions.append({"feature": k, "label": labels[k], "value": values[k],
                                      "typical": b["reference"][k], "delta": delta})
    contributions.sort(key=lambda c: -abs(c["delta"]))

    band, band_label = risk_band(prob, b["threshold"])
    return {
        "probability": prob, "threshold": b["threshold"], "positive": prob >= b["threshold"],
        "band": band, "band_label": band_label, "model_name": b["model_name"],
        "contributions": contributions[:6], "missing": [labels[k] for k in feats if values.get(k) is None],
    }
