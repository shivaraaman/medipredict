"""Architecture / workflow diagrams for the project report. Run: python reports/diagrams.py"""
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from ml.plots import INK, INK2, SURFACE  # noqa: E402,F401

OUT = Path(__file__).resolve().parent / "figures"
TEAL, TEAL_SOFT = "#0e5c56", "#e2efec"
BLUE_SOFT, ORANGE_SOFT, GREY_SOFT = "#e6f0fc", "#fdeee7", "#f1f0ec"


def box(ax, x, y, w, h, title, sub="", fc=TEAL_SOFT, ec=TEAL, fs=10.5):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.08",
                                fc=fc, ec=ec, lw=1.3))
    if sub:
        ax.text(x + w / 2, y + h * 0.63, title, ha="center", va="center", fontsize=fs, fontweight="bold", color=INK)
        ax.text(x + w / 2, y + h * 0.3, sub, ha="center", va="center", fontsize=fs - 2.3, color=INK2, linespacing=1.3)
    else:
        ax.text(x + w / 2, y + h / 2, title, ha="center", va="center", fontsize=fs, fontweight="bold", color=INK)


def arrow(ax, a, b, label="", rad=0.0):
    ax.add_patch(FancyArrowPatch(a, b, arrowstyle="-|>", mutation_scale=13, lw=1.3, color=INK2,
                                 connectionstyle=f"arc3,rad={rad}"))
    if label:
        ax.text((a[0] + b[0]) / 2, (a[1] + b[1]) / 2 + 0.12, label, ha="center", fontsize=8, color=INK2,
                bbox=dict(fc=SURFACE, ec="none", pad=1))


def canvas(w, h):
    fig, ax = plt.subplots(figsize=(w, h))
    fig.patch.set_facecolor("white")
    ax.set_xlim(0, w)
    ax.set_ylim(0, h)
    ax.axis("off")
    return fig, ax


def save(fig, name):
    fig.savefig(OUT / name, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def architecture():
    fig, ax = canvas(12, 6.6)
    # tiers
    for y, h, lab in [(4.75, 1.55, "Presentation layer"), (2.45, 2.05, "Application layer"), (0.25, 1.95, "Data & model layer")]:
        ax.add_patch(FancyBboxPatch((0.15, y), 11.7, h, boxstyle="round,pad=0.02,rounding_size=0.1",
                                    fc=GREY_SOFT, ec="none"))
        ax.text(0.35, y + h - 0.22, lab.upper(), fontsize=8, color=INK2, fontweight="bold")
    box(ax, 0.6, 4.95, 3.0, 0.95, "Clinician / user", "web browser (desktop or mobile)", BLUE_SOFT, "#2a78d6")
    box(ax, 4.5, 4.95, 3.0, 0.95, "Web pages (HTML/CSS)", "Jinja2 templates: forms, results,\nperformance, history", BLUE_SOFT, "#2a78d6")
    box(ax, 8.4, 4.95, 3.0, 0.95, "External systems", "JSON REST API client", BLUE_SOFT, "#2a78d6")
    box(ax, 0.6, 2.7, 3.0, 1.3, "Flask application", "routing, sessions,\ninput validation\n(gunicorn WSGI server)")
    box(ax, 4.5, 2.7, 3.0, 1.3, "Prediction engine", "preprocessing pipeline →\nmodel → probability →\nrisk band & threshold")
    box(ax, 8.4, 2.7, 3.0, 1.3, "Explanation module", "per-patient contribution\nanalysis (what drove\nthe result)")
    box(ax, 0.6, 0.45, 3.0, 1.2, "SQLite database", "assessment history\n(no personal names)", ORANGE_SOFT, "#eb6834")
    box(ax, 4.5, 0.45, 3.0, 1.2, "Trained models (.joblib)", "Diabetes: ANN · Heart: LR\nCKD: ANN", ORANGE_SOFT, "#eb6834")
    box(ax, 8.4, 0.45, 3.0, 1.2, "Offline training pipeline", "3 datasets · 7 algorithms\n5-fold CV · metrics & figures", ORANGE_SOFT, "#eb6834")
    arrow(ax, (3.6, 5.42), (4.5, 5.42), "HTTPS")
    arrow(ax, (5.2, 4.95), (3.0, 4.0), "form POST")
    arrow(ax, (9.0, 4.95), (3.5, 4.0), "JSON API")
    arrow(ax, (3.6, 3.35), (4.5, 3.35))
    arrow(ax, (7.5, 3.35), (8.4, 3.35))
    arrow(ax, (2.1, 2.7), (2.1, 1.65), "save / read")
    arrow(ax, (6.0, 1.65), (6.0, 2.7), "load")
    arrow(ax, (8.4, 1.05), (7.5, 1.05), "export")
    save(fig, "system_architecture.png")


def pipeline():
    fig, ax = canvas(14.2, 3.2)
    steps = [("Raw data", "Pima · Cleveland\n· UCI CKD"), ("Cleaning", "impossible zeros,\n'?' → missing"),
             ("Stratified split", "80% train\n20% test"), ("Preprocessing", "impute · scale\n· one-hot"),
             ("Grid search", "7 models ×\n5-fold CV (AUC)"), ("Threshold tuning", "Youden's J on\nout-of-fold preds"),
             ("Test evaluation", "acc · prec · recall\n· spec · F1 · AUC"), ("Deployment", "best model\n→ .joblib → web")]
    w, gap = 1.58, 0.17
    for i, (t, s) in enumerate(steps):
        x = 0.15 + i * (w + gap)
        box(ax, x, 0.9, w, 1.5, t, s, BLUE_SOFT if i < 4 else TEAL_SOFT, "#2a78d6" if i < 4 else TEAL, fs=9.2)
        if i:
            arrow(ax, (x - gap, 1.65), (x, 1.65))
    ax.text(0.15, 2.75, "Data preparation", fontsize=9, color="#2a78d6", fontweight="bold")
    ax.text(0.15 + 4 * (w + gap), 2.75, "Modelling & deployment", fontsize=9, color=TEAL, fontweight="bold")
    save(fig, "ml_pipeline.png")


def ann(layers, title, name):
    fig, ax = canvas(10, 5.4)
    shown = [min(n, 8) for n in layers]
    xs = [1.2 + i * (7.6 / (len(layers) - 1)) for i in range(len(layers))]
    pos = []
    for x, n_show, n in zip(xs, shown, layers):
        ys = [2.7 + (j - (n_show - 1) / 2) * 0.52 for j in range(n_show)]
        pos.append([(x, y) for y in ys])
    for a, b in zip(pos[:-1], pos[1:]):
        for p in a:
            for q in b:
                ax.plot([p[0], q[0]], [p[1], q[1]], color="#c9c7c0", lw=0.45, zorder=1)
    labels = ["Input layer"] + [f"Hidden layer {i}" for i in range(1, len(layers) - 1)] + ["Output layer"]
    for i, (pts, n, lab) in enumerate(zip(pos, layers, labels)):
        col = "#2a78d6" if i == 0 else ("#eb6834" if i == len(layers) - 1 else TEAL)
        for j, (x, y) in enumerate(pts):
            if n > 8 and j == 4:
                ax.text(x, y, "⋮", ha="center", va="center", fontsize=14, color=INK2, zorder=3)
                continue
            ax.add_patch(plt.Circle((x, y), 0.16, fc="white", ec=col, lw=1.6, zorder=2))
        sub = f"{n} features" if i == 0 else ("1 neuron · sigmoid" if i == len(layers) - 1 else f"{n} neurons · ReLU")
        ax.text(pts[0][0], 0.35, lab, ha="center", fontsize=9.5, fontweight="bold", color=INK)
        ax.text(pts[0][0], 0.05, sub, ha="center", fontsize=8.5, color=INK2)
    ax.text(xs[-1] + 0.45, 2.7, "P(disease)", va="center", fontsize=9, color=INK2)
    ax.text(0.1, 5.15, title, fontsize=11.5, fontweight="bold", color=INK)
    save(fig, name)


def database():
    fig, ax = canvas(5.6, 4.0)
    rows = [("id", "TEXT  PK"), ("created_at", "TEXT"), ("disease", "TEXT"), ("inputs", "TEXT (JSON)"),
            ("probability", "REAL"), ("band", "TEXT"), ("model_name", "TEXT"), ("patient_ref", "TEXT NULL")]
    ax.add_patch(FancyBboxPatch((0.4, 0.2), 4.8, 3.6, boxstyle="round,pad=0.02,rounding_size=0.08",
                                fc="white", ec=TEAL, lw=1.3))
    ax.add_patch(FancyBboxPatch((0.4, 3.25), 4.8, 0.55, boxstyle="round,pad=0.02,rounding_size=0.08",
                                fc=TEAL, ec=TEAL))
    ax.text(2.8, 3.52, "assessments", ha="center", va="center", color="white", fontweight="bold", fontsize=11)
    for i, (c, t) in enumerate(rows):
        y = 2.95 - i * 0.36
        ax.text(0.65, y, c, fontsize=9.5, color=INK, family="monospace", va="center")
        ax.text(4.95, y, t, fontsize=9, color=INK2, ha="right", va="center")
    save(fig, "database_schema.png")


if __name__ == "__main__":
    import json
    m = json.loads((OUT.parent / "metrics.json").read_text())
    architecture()
    pipeline()
    ann(m["diabetes"]["ann"]["layers"], "ANN for diabetes: 8 → 64 → 32 → 1  (2,689 trainable parameters)", "ann_diabetes.png")
    ann(m["kidney"]["ann"]["layers"], "ANN for chronic kidney disease: 24 → 32 → 16 → 1  (1,345 parameters)", "ann_kidney.png")
    database()
    print("diagrams written to", OUT)
