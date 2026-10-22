"""
Disease definitions shared by the training pipeline and the web application.

Each disease has:
  * a loader that returns (X: DataFrame, y: Series) from the raw dataset
  * a list of feature specifications used to build preprocessing and web forms
"""
from pathlib import Path

import numpy as np
import pandas as pd

DATA_DIR = Path(__file__).resolve().parent.parent / "data"


def num(key, label, unit="", lo=None, hi=None, step="any", help="", required=True, example=None):
    return dict(key=key, label=label, kind="number", unit=unit, min=lo, max=hi,
                step=step, help=help, required=required, example=example)


def cat(key, label, options, help="", required=True, example=None):
    """options: list of (value, display label)."""
    return dict(key=key, label=label, kind="select", options=options,
                help=help, required=required, example=example)


# --------------------------------------------------------------------------
# 1. Diabetes  (Pima Indians Diabetes Database, NIDDK / UCI, n = 768)
# --------------------------------------------------------------------------
DIABETES_FEATURES = [
    num("Pregnancies", "Number of pregnancies", "", 0, 20, 1, "Enter 0 if not applicable", example=2),
    num("Glucose", "Plasma glucose (2-h OGTT)", "mg/dL", 40, 250, 1,
        "Glucose concentration 2 hours after an oral glucose tolerance test", example=120),
    num("BloodPressure", "Diastolic blood pressure", "mm Hg", 30, 140, 1, example=72),
    num("SkinThickness", "Triceps skin-fold thickness", "mm", 5, 100, 1, required=False, example=25),
    num("Insulin", "2-hour serum insulin", "µU/mL", 10, 900, 1, required=False, example=80),
    num("BMI", "Body mass index", "kg/m²", 10, 70, 0.1, example=28.5),
    num("DiabetesPedigreeFunction", "Diabetes pedigree function", "", 0.05, 2.5, 0.001,
        "Score for family history of diabetes (0.08 – 2.42). Use 0.47 if unknown", example=0.47),
    num("Age", "Age", "years", 18, 100, 1, example=35),
]


def load_diabetes():
    cols = [f["key"] for f in DIABETES_FEATURES] + ["Outcome"]
    df = pd.read_csv(DATA_DIR / "diabetes.csv", header=None, names=cols)
    # A value of 0 is physiologically impossible for these measurements -> missing.
    for c in ["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"]:
        df[c] = df[c].replace(0, np.nan)
    return df.drop(columns="Outcome"), df["Outcome"].astype(int)


# --------------------------------------------------------------------------
# 2. Heart disease  (UCI Cleveland Heart Disease, n = 303)
# --------------------------------------------------------------------------
HEART_FEATURES = [
    num("age", "Age", "years", 18, 100, 1, example=54),
    cat("sex", "Sex", [(1, "Male"), (0, "Female")], example=1),
    cat("cp", "Chest pain type", [(1, "Typical angina"), (2, "Atypical angina"),
                                   (3, "Non-anginal pain"), (4, "Asymptomatic")], example=3),
    num("trestbps", "Resting blood pressure", "mm Hg", 80, 220, 1, example=130),
    num("chol", "Serum cholesterol", "mg/dL", 100, 600, 1, example=240),
    cat("fbs", "Fasting blood sugar > 120 mg/dL", [(0, "No"), (1, "Yes")], example=0),
    cat("restecg", "Resting ECG result", [(0, "Normal"), (1, "ST-T wave abnormality"),
                                          (2, "Left ventricular hypertrophy")], example=0),
    num("thalach", "Maximum heart rate achieved", "bpm", 60, 220, 1, example=150),
    cat("exang", "Exercise-induced angina", [(0, "No"), (1, "Yes")], example=0),
    num("oldpeak", "ST depression (exercise vs rest)", "mm", 0, 7, 0.1, example=1.0),
    cat("slope", "Slope of peak exercise ST segment",
        [(1, "Upsloping"), (2, "Flat"), (3, "Downsloping")], example=1),
    cat("ca", "Major vessels coloured by fluoroscopy", [(0, "0"), (1, "1"), (2, "2"), (3, "3")],
        required=False, example=0),
    cat("thal", "Thallium stress test", [(3, "Normal"), (6, "Fixed defect"), (7, "Reversible defect")],
        required=False, example=3),
]
HEART_CATEGORICAL = ["cp", "restecg", "slope", "thal"]  # nominal -> one-hot


def load_heart():
    cols = [f["key"] for f in HEART_FEATURES] + ["num"]
    df = pd.read_csv(DATA_DIR / "heart_cleveland.data", header=None, names=cols, na_values="?")
    y = (df["num"] > 0).astype(int)  # 0 = no disease, 1-4 = disease present
    return df.drop(columns="num"), y


# --------------------------------------------------------------------------
# 3. Chronic kidney disease  (UCI CKD, Apollo Hospitals, n = 400)
# --------------------------------------------------------------------------
YN = [(0, "No"), (1, "Yes")]
KIDNEY_FEATURES = [
    num("age", "Age", "years", 1, 100, 1, example=48),
    num("bp", "Blood pressure (diastolic)", "mm Hg", 40, 180, 1, example=80),
    cat("sg", "Urine specific gravity", [(1.005, "1.005"), (1.010, "1.010"), (1.015, "1.015"),
                                         (1.020, "1.020"), (1.025, "1.025")], example=1.020),
    cat("al", "Urine albumin (0–5)", [(i, str(i)) for i in range(6)], example=0),
    cat("su", "Urine sugar (0–5)", [(i, str(i)) for i in range(6)], example=0),
    cat("rbc", "Red blood cells in urine", [(0, "Normal"), (1, "Abnormal")], required=False, example=0),
    cat("pc", "Pus cells in urine", [(0, "Normal"), (1, "Abnormal")], required=False, example=0),
    cat("pcc", "Pus cell clumps", [(0, "Not present"), (1, "Present")], required=False, example=0),
    cat("ba", "Bacteria in urine", [(0, "Not present"), (1, "Present")], required=False, example=0),
    num("bgr", "Random blood glucose", "mg/dL", 20, 500, 1, required=False, example=121),
    num("bu", "Blood urea", "mg/dL", 1, 400, 0.1, required=False, example=36),
    num("sc", "Serum creatinine", "mg/dL", 0.2, 80, 0.1, example=1.2),
    num("sod", "Sodium", "mEq/L", 100, 170, 0.1, required=False, example=138),
    num("pot", "Potassium", "mEq/L", 2, 50, 0.1, required=False, example=4.4),
    num("hemo", "Haemoglobin", "g/dL", 3, 20, 0.1, example=15.4),
    num("pcv", "Packed cell volume", "%", 9, 60, 1, required=False, example=44),
    num("wbcc", "White blood cell count", "cells/cumm", 2000, 27000, 100, required=False, example=7800),
    num("rbcc", "Red blood cell count", "millions/cmm", 2, 8, 0.1, required=False, example=5.2),
    cat("htn", "Hypertension", YN, example=0),
    cat("dm", "Diabetes mellitus", YN, example=0),
    cat("cad", "Coronary artery disease", YN, required=False, example=0),
    cat("appet", "Appetite", [(0, "Good"), (1, "Poor")], required=False, example=0),
    cat("pe", "Pedal (ankle) oedema", YN, required=False, example=0),
    cat("ane", "Anaemia", YN, required=False, example=0),
]
_KIDNEY_MAP = {
    "rbc": {"normal": 0, "abnormal": 1}, "pc": {"normal": 0, "abnormal": 1},
    "pcc": {"notpresent": 0, "present": 1}, "ba": {"notpresent": 0, "present": 1},
    "htn": {"no": 0, "yes": 1}, "dm": {"no": 0, "yes": 1}, "cad": {"no": 0, "yes": 1},
    "appet": {"good": 0, "poor": 1}, "pe": {"no": 0, "yes": 1}, "ane": {"no": 0, "yes": 1},
}


def load_kidney():
    df = pd.read_csv(DATA_DIR / "kidney.csv", dtype=str)
    df = df.apply(lambda s: s.str.strip()).replace("?", np.nan)
    for c, m in _KIDNEY_MAP.items():
        df[c] = df[c].map(m)
    y = (df.pop("class") == "ckd").astype(int)
    df = df.apply(pd.to_numeric, errors="coerce")
    return df[[f["key"] for f in KIDNEY_FEATURES]], y


DISEASES = {
    "diabetes": dict(
        name="Diabetes", icon="drop", loader=load_diabetes, features=DIABETES_FEATURES,
        onehot=[], positive="Diabetic", negative="Non-diabetic",
        dataset="Pima Indians Diabetes Database (NIDDK, via UCI ML Repository)",
    ),
    "heart": dict(
        name="Heart Disease", icon="heart", loader=load_heart, features=HEART_FEATURES,
        onehot=HEART_CATEGORICAL, positive="Heart disease", negative="No heart disease",
        dataset="UCI Heart Disease — Cleveland Clinic Foundation subset",
    ),
    "kidney": dict(
        name="Chronic Kidney Disease", icon="kidney", loader=load_kidney, features=KIDNEY_FEATURES,
        onehot=[], positive="CKD", negative="No CKD",
        dataset="UCI Chronic Kidney Disease dataset (Apollo Hospitals, Tamil Nadu)",
    ),
}


def feature_keys(disease):
    return [f["key"] for f in DISEASES[disease]["features"]]


def build_preprocessor(disease):
    """Median-impute + standardise numeric columns; mode-impute + one-hot nominal columns."""
    from sklearn.compose import ColumnTransformer
    from sklearn.impute import SimpleImputer
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import OneHotEncoder, StandardScaler

    spec = DISEASES[disease]
    onehot = spec["onehot"]
    numeric = [k for k in feature_keys(disease) if k not in onehot]
    transformers = [("num", Pipeline([("impute", SimpleImputer(strategy="median")),
                                      ("scale", StandardScaler())]), numeric)]
    if onehot:
        transformers.append(("cat", Pipeline([("impute", SimpleImputer(strategy="most_frequent")),
                                              ("onehot", OneHotEncoder(handle_unknown="ignore"))]), onehot))
    return ColumnTransformer(transformers)
