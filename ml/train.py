"""
MediPredict — training & evaluation pipeline.

For each disease (diabetes, heart, kidney):
  1. load + clean the raw dataset
  2. stratified 80/20 train/test split (random_state=42)
  3. tune 7 classifiers with GridSearchCV (5-fold stratified CV, ROC-AUC)
  4. evaluate every tuned model on the untouched test set
  5. pick the deployed model by *cross-validated* ROC-AUC (never by test score)
  6. save model bundle (joblib), metrics.json and all report figures

Run:  python -m ml.train
"""
import json
import time
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score, precision_score,
                             recall_score, roc_auc_score, roc_curve)
from sklearn.model_selection import (GridSearchCV, StratifiedKFold, cross_val_predict, cross_validate,
                                     train_test_split)
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

from ml import plots
from ml.diseases import DISEASES, build_preprocessor, feature_keys

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parent.parent
MODEL_DIR = ROOT / "models"
REPORT_DIR = ROOT / "reports"
FIG_DIR = REPORT_DIR / "figures"
SEED = 42

# Order matters: it fixes the colour of each model in every chart.
MODELS = {
    "Logistic Regression": (LogisticRegression(max_iter=5000, random_state=SEED),
                            {"clf__C": [0.01, 0.1, 1, 10]}),
    "K-Nearest Neighbours": (KNeighborsClassifier(),
                             {"clf__n_neighbors": [3, 5, 7, 9, 11, 15], "clf__weights": ["uniform", "distance"]}),
    "Support Vector Machine": (SVC(probability=True, random_state=SEED),
                               {"clf__C": [0.1, 1, 10], "clf__gamma": ["scale", 0.01, 0.1]}),
    "Decision Tree": (DecisionTreeClassifier(random_state=SEED),
                      {"clf__max_depth": [3, 4, 5, 6, None], "clf__min_samples_leaf": [1, 5, 10]}),
    "Random Forest": (RandomForestClassifier(n_estimators=300, random_state=SEED, n_jobs=-1),
                      {"clf__max_depth": [None, 5, 8], "clf__min_samples_leaf": [1, 3, 5]}),
    "Gradient Boosting": (GradientBoostingClassifier(random_state=SEED),
                          {"clf__n_estimators": [100, 200], "clf__learning_rate": [0.05, 0.1],
                           "clf__max_depth": [2, 3]}),
    "Artificial Neural Network (MLP)": (
        MLPClassifier(activation="relu", solver="adam", learning_rate_init=1e-3, max_iter=2000,
                      early_stopping=True, validation_fraction=0.15, n_iter_no_change=30,
                      random_state=SEED),
        {"clf__hidden_layer_sizes": [(16,), (32, 16), (64, 32), (64, 32, 16)],
         "clf__alpha": [1e-4, 1e-3, 1e-2, 1e-1]}),
}
ANN = "Artificial Neural Network (MLP)"


def metrics_at(y_true, prob, thr=0.5):
    pred = (prob >= thr).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, pred, labels=[0, 1]).ravel()
    return {
        "accuracy": accuracy_score(y_true, pred),
        "precision": precision_score(y_true, pred, zero_division=0),
        "recall": recall_score(y_true, pred, zero_division=0),
        "specificity": tn / (tn + fp) if (tn + fp) else 0.0,
        "f1": f1_score(y_true, pred, zero_division=0),
        "roc_auc": roc_auc_score(y_true, prob),
        "confusion_matrix": [[int(tn), int(fp)], [int(fn), int(tp)]],
    }


def ann_details(pipe):
    mlp = pipe.named_steps["clf"]
    n_in = mlp.coefs_[0].shape[0]
    layers = [n_in] + list(mlp.hidden_layer_sizes) + [1]
    return {
        "layers": layers,
        "activation": mlp.activation,
        "output_activation": mlp.out_activation_,
        "optimizer": "Adam",
        "learning_rate": mlp.learning_rate_init,
        "l2_alpha": mlp.alpha,
        "epochs_trained": int(mlp.n_iter_),
        "trainable_parameters": int(sum(w.size for w in mlp.coefs_) + sum(b.size for b in mlp.intercepts_)),
        "loss_curve": [float(v) for v in mlp.loss_curve_],
        "val_accuracy_curve": [float(v) for v in (mlp.validation_scores_ or [])],
    }


def train_disease(key):
    spec = DISEASES[key]
    X, y = spec["loader"]()
    feats = feature_keys(key)
    X = X[feats]
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, stratify=y, random_state=SEED)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)

    print(f"\n=== {spec['name']}: {len(X)} records, {X.shape[1]} features, "
          f"positive rate {y.mean():.1%}, missing cells {int(X.isna().sum().sum())}")
    results, fitted, curves = {}, {}, {}
    for name, (est, grid) in MODELS.items():
        t0 = time.time()
        pipe = Pipeline([("prep", build_preprocessor(key)), ("clf", clone(est))])
        gs = GridSearchCV(pipe, grid, cv=cv, scoring="roc_auc", n_jobs=-1).fit(X_tr, y_tr)
        best = gs.best_estimator_
        cvr = cross_validate(clone(best), X_tr, y_tr, cv=cv,
                             scoring=["accuracy", "f1", "roc_auc", "recall", "precision"], n_jobs=-1)
        # Decision threshold chosen on out-of-fold training predictions (Youden's J = sens + spec - 1)
        oof = cross_val_predict(clone(best), X_tr, y_tr, cv=cv, method="predict_proba", n_jobs=-1)[:, 1]
        f, t, thr = roc_curve(y_tr, oof)
        threshold = float(np.clip(thr[np.argmax(t - f)], 0.05, 0.95))
        prob = best.predict_proba(X_te)[:, 1]
        m = metrics_at(y_te, prob, threshold)
        m.update({
            "threshold": round(threshold, 4), "default_threshold_metrics": metrics_at(y_te, prob, 0.5),
            "cv_accuracy_mean": cvr["test_accuracy"].mean(), "cv_accuracy_std": cvr["test_accuracy"].std(),
            "cv_f1_mean": cvr["test_f1"].mean(), "cv_roc_auc_mean": cvr["test_roc_auc"].mean(),
            "cv_roc_auc_std": cvr["test_roc_auc"].std(),
            "best_params": {k.replace("clf__", ""): (list(v) if isinstance(v, tuple) else v)
                            for k, v in gs.best_params_.items()},
            "train_seconds": round(time.time() - t0, 2),
        })
        results[name], fitted[name] = m, best
        fpr, tpr, _ = roc_curve(y_te, prob)
        curves[name] = (fpr, tpr)
        print(f"  {name:34s} CV-AUC {m['cv_roc_auc_mean']:.3f}  test acc {m['accuracy']:.3f} "
              f"F1 {m['f1']:.3f} rec {m['recall']:.3f} thr {threshold:.2f} AUC {m['roc_auc']:.3f}  {m['best_params']}")

    # Model selection on cross-validation only (test set stays unbiased).
    # Prefer the ANN when it is within 0.005 CV-AUC of the leader (simpler story, same quality).
    leader = max(results, key=lambda n: results[n]["cv_roc_auc_mean"])
    chosen = ANN if results[leader]["cv_roc_auc_mean"] - results[ANN]["cv_roc_auc_mean"] <= 0.005 else leader
    model = fitted[chosen]
    print(f"  -> deployed: {chosen}")

    # Global explanation: permutation importance (drop in ROC-AUC) on the test set.
    pi = permutation_importance(model, X_te, y_te, scoring="roc_auc", n_repeats=30, random_state=SEED, n_jobs=-1)
    importance = sorted(({"feature": f, "label": s["label"], "importance": float(m_), "std": float(sd)}
                         for f, s, m_, sd in zip(feats, spec["features"], pi.importances_mean, pi.importances_std)),
                        key=lambda d: -d["importance"])

    # Reference values for local (per-patient) explanations in the web app.
    reference = {c: float(X_tr[c].median()) if c not in spec["onehot"] else float(X_tr[c].mode()[0])
                 for c in feats}

    bundle = {"disease": key, "model_name": chosen, "pipeline": model, "features": feats,
              "reference": reference, "threshold": results[chosen]["threshold"], "metrics": results[chosen],
              "trained_at": time.strftime("%Y-%m-%d %H:%M"), "sklearn": __import__("sklearn").__version__}
    joblib.dump(bundle, MODEL_DIR / f"{key}.joblib", compress=3)

    # Figures
    plots.roc_curves(curves, results, spec["name"], FIG_DIR / f"{key}_roc.png")
    plots.model_comparison(results, spec["name"], FIG_DIR / f"{key}_comparison.png")
    plots.confusion(results[chosen]["confusion_matrix"], spec, chosen, FIG_DIR / f"{key}_confusion.png")
    if chosen != ANN:
        plots.confusion(results[ANN]["confusion_matrix"], spec, ANN, FIG_DIR / f"{key}_confusion_ann.png")
    plots.ann_curves(ann_details(fitted[ANN]), spec["name"], FIG_DIR / f"{key}_ann_training.png")
    plots.importance(importance, spec["name"], FIG_DIR / f"{key}_importance.png")
    plots.correlation(X.assign(Target=y), spec["name"], FIG_DIR / f"{key}_correlation.png")

    return {
        "name": spec["name"], "dataset": spec["dataset"], "n_records": int(len(X)),
        "n_features": int(X.shape[1]), "n_positive": int(y.sum()), "n_negative": int((1 - y).sum()),
        "missing_cells": int(X.isna().sum().sum()),
        "missing_by_feature": {c: int(v) for c, v in X.isna().sum().items() if v},
        "n_train": int(len(X_tr)), "n_test": int(len(X_te)),
        "deployed_model": chosen, "results": results, "ann": ann_details(fitted[ANN]),
        "importance": importance,
        "describe": X.describe().T.round(2).reset_index().rename(columns={"index": "feature"}).to_dict("records"),
    }


def main():
    MODEL_DIR.mkdir(exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    report = {k: train_disease(k) for k in DISEASES}
    plots.class_balance(report, FIG_DIR / "class_balance.png")
    with open(REPORT_DIR / "metrics.json", "w") as f:
        json.dump(report, f, indent=2, default=lambda o: o.item() if hasattr(o, "item") else str(o))
    print("\nSaved models to", MODEL_DIR, "and metrics/figures to", REPORT_DIR)


if __name__ == "__main__":
    main()
