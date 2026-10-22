"""Report figures (matplotlib). Palette: validated categorical order, light surface."""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK2 = "#52514e"
GRID = "#e4e3df"
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
SEQ = LinearSegmentedColormap.from_list("seq", ["#f4f8fd", "#9ec5f4", "#2a78d6", "#104281"])
DIV = LinearSegmentedColormap.from_list("div", ["#2a78d6", "#f0efec", "#eb6834"])

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "axes.edgecolor": GRID, "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
    "text.color": INK, "axes.titlecolor": INK, "axes.titleweight": "bold", "axes.titlesize": 12,
    "axes.titlelocation": "left", "font.size": 10, "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6, "legend.frameon": False,
    "font.family": "DejaVu Sans", "savefig.dpi": 200, "savefig.bbox": "tight",
})

SHORT = {"Logistic Regression": "Logistic Reg.", "K-Nearest Neighbours": "KNN",
         "Support Vector Machine": "SVM", "Decision Tree": "Decision Tree",
         "Random Forest": "Random Forest", "Gradient Boosting": "Grad. Boosting",
         "Artificial Neural Network (MLP)": "ANN (MLP)"}


def _save(fig, path):
    fig.savefig(path)
    plt.close(fig)


def roc_curves(curves, results, title, path):
    fig, ax = plt.subplots(figsize=(6.4, 5.2))
    ax.plot([0, 1], [0, 1], ls="--", lw=1, color=INK2, alpha=0.6, label="Chance (AUC 0.50)")
    for i, (name, (fpr, tpr)) in enumerate(curves.items()):
        ax.plot(fpr, tpr, lw=2, color=SERIES[i],
                label=f"{SHORT[name]} (AUC {results[name]['roc_auc']:.3f})")
    ax.set(xlim=(-0.01, 1.01), ylim=(-0.01, 1.01), xlabel="False positive rate (1 − specificity)",
           ylabel="True positive rate (sensitivity)")
    ax.set_title(f"ROC curves on test set — {title}")
    ax.legend(loc="lower right", fontsize=8.5)
    _save(fig, path)


def model_comparison(results, title, path):
    names = list(results)
    metrics = [("accuracy", "Accuracy"), ("f1", "F1-score"), ("roc_auc", "ROC-AUC")]
    y = np.arange(len(names))
    h = 0.26
    fig, ax = plt.subplots(figsize=(7.6, 5.0))
    for j, (k, lab) in enumerate(metrics):
        vals = [results[n][k] for n in names]
        ax.barh(y + (j - 1) * h, vals, height=h - 0.03, color=SERIES[j], label=lab, zorder=3)
        for yi, v in zip(y, vals):
            ax.text(v + 0.005, yi + (j - 1) * h, f"{v:.3f}", va="center", fontsize=7, color=INK2)
    ax.set_yticks(y, [SHORT[n] for n in names])
    ax.invert_yaxis()
    lo = min(min(results[n][k] for n in names) for k, _ in metrics)
    ax.set_xlim(max(0, lo - 0.08), 1.04)
    ax.grid(axis="y", visible=False)
    ax.set_xlabel("Score on held-out test set")
    ax.set_title(f"Model comparison — {title}")
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=3, fontsize=8.5, borderaxespad=1.6)
    _save(fig, path)


def confusion(cm, spec, model_name, path):
    cm = np.array(cm)
    fig, ax = plt.subplots(figsize=(4.4, 3.9))
    ax.imshow(cm, cmap=SEQ)
    ax.grid(False)
    for i in range(2):
        for j in range(2):
            ax.text(j, i, cm[i, j], ha="center", va="center", fontsize=16, fontweight="bold",
                    color="white" if cm[i, j] > cm.max() * 0.55 else INK)
    labs = [spec["negative"], spec["positive"]]
    ax.set_xticks([0, 1], labs)
    ax.set_yticks([0, 1], labs, rotation=90, va="center")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title(f"{SHORT.get(model_name, model_name)} — {spec['name']}", fontsize=11)
    for s in ax.spines.values():
        s.set_visible(False)
    _save(fig, path)


def ann_curves(ann, title, path):
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(9, 3.6))
    ep = np.arange(1, len(ann["loss_curve"]) + 1)
    a1.plot(ep, ann["loss_curve"], color=SERIES[0], lw=2)
    a1.set(xlabel="Epoch", ylabel="Binary cross-entropy loss")
    a1.set_title("Training loss")
    if ann["val_accuracy_curve"]:
        a2.plot(ep[:len(ann["val_accuracy_curve"])], ann["val_accuracy_curve"], color=SERIES[1], lw=2)
    a2.set(xlabel="Epoch", ylabel="Accuracy")
    a2.set_title("Validation accuracy (15% hold-out)")
    fig.suptitle(f"ANN training history — {title}", x=0.01, ha="left", fontweight="bold", color=INK)
    fig.tight_layout()
    _save(fig, path)


def importance(imp, title, path, top=12):
    imp = imp[:top][::-1]
    fig, ax = plt.subplots(figsize=(7, 0.38 * len(imp) + 1.2))
    vals = [d["importance"] for d in imp]
    ax.barh(range(len(imp)), vals, xerr=[d["std"] for d in imp], color=SERIES[0], height=0.6,
            error_kw=dict(ecolor=INK2, lw=0.8, capsize=2), zorder=3)
    ax.set_yticks(range(len(imp)), [d["label"] for d in imp], fontsize=8.5)
    ax.grid(axis="y", visible=False)
    ax.axvline(0, color=INK2, lw=0.8)
    ax.set_xlabel("Mean drop in ROC-AUC when feature is shuffled")
    ax.set_title(f"Permutation feature importance — {title}")
    _save(fig, path)


def correlation(df, title, path):
    c = df.corr(numeric_only=True)
    n = len(c)
    fig, ax = plt.subplots(figsize=(0.42 * n + 2.4, 0.42 * n + 1.6))
    im = ax.imshow(c, cmap=DIV, vmin=-1, vmax=1)
    ax.grid(False)
    ax.set_xticks(range(n), c.columns, rotation=90, fontsize=8)
    ax.set_yticks(range(n), c.columns, fontsize=8)
    if n <= 15:
        for i in range(n):
            for j in range(n):
                ax.text(j, i, f"{c.iat[i, j]:.2f}", ha="center", va="center", fontsize=6.5, color=INK)
    for s in ax.spines.values():
        s.set_visible(False)
    fig.colorbar(im, ax=ax, fraction=0.04, pad=0.02).outline.set_visible(False)
    ax.set_title(f"Pearson correlation — {title}")
    _save(fig, path)


def class_balance(report, path):
    fig, axes = plt.subplots(1, 3, figsize=(10, 3.2), sharey=False)
    for ax, (k, r) in zip(axes, report.items()):
        vals = [r["n_negative"], r["n_positive"]]
        bars = ax.bar(["Negative", "Positive"], vals, color=[SERIES[0], SERIES[1]], width=0.55, zorder=3)
        for b, v in zip(bars, vals):
            ax.text(b.get_x() + b.get_width() / 2, v, f"{v} ({v / sum(vals):.0%})", ha="center",
                    va="bottom", fontsize=9, color=INK2)
        ax.set_title(r["name"], fontsize=11)
        ax.grid(axis="x", visible=False)
        ax.set_ylim(0, max(vals) * 1.18)
    axes[0].set_ylabel("Patients")
    fig.suptitle("Class distribution of the three datasets", x=0.01, ha="left", fontweight="bold")
    fig.tight_layout()
    _save(fig, path)
