"""
compare.py — Side-by-side comparison of the NumPy and sklearn pipelines.

Runs both implementations on the same Iris data, prints a comparison table,
and saves a four-panel figure (loss curve, accuracy bars, two confusion matrices)
to plots/comparison.png.

Run
---
    python compare.py
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression as SklearnLR
from sklearn.preprocessing import StandardScaler as SklearnScaler
from sklearn.model_selection import train_test_split as sklearn_tts
from sklearn.metrics import (
    accuracy_score    as sk_accuracy,
    confusion_matrix  as sk_confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)

from numpy_lr import (
    SoftmaxRegression,
    StandardScaler    as NumpyScaler,
    train_test_split  as numpy_tts,
    accuracy_score    as np_accuracy,
    confusion_matrix  as np_confusion_matrix,
)

SEED        = 42
PLOT_DIR    = "plots"
CLASS_NAMES = ["Setosa", "Versicolor", "Virginica"]
os.makedirs(PLOT_DIR, exist_ok=True)


# ── Run NumPy pipeline ─────────────────────────────────────────────────────────

def run_numpy(X, y):
    X_tr, X_te, y_tr, y_te = numpy_tts(X, y, test_size=0.2, random_state=SEED)
    scaler = NumpyScaler()
    X_tr   = scaler.fit_transform(X_tr)
    X_te   = scaler.transform(X_te)

    model = SoftmaxRegression(learning_rate=0.5, n_iters=2000, random_state=SEED)
    model.fit(X_tr, y_tr)

    y_pred_tr = model.predict(X_tr)
    y_pred_te = model.predict(X_te)

    return {
        "name":        "NumPy\n(gradient descent)",
        "losses":      model.losses,
        "train_acc":   np_accuracy(y_tr, y_pred_tr),
        "test_acc":    np_accuracy(y_te, y_pred_te),
        "cm":          np_confusion_matrix(y_te, y_pred_te, n_classes=3),
        "macro_f1":    _macro_f1_numpy(y_te, y_pred_te),
        "macro_prec":  _macro_prec_numpy(y_te, y_pred_te),
        "macro_rec":   _macro_rec_numpy(y_te, y_pred_te),
        "y_te":        y_te,
        "y_pred":      y_pred_te,
    }


def run_sklearn(X, y):
    X_tr, X_te, y_tr, y_te = sklearn_tts(
        X, y, test_size=0.2, random_state=SEED, stratify=y
    )
    scaler = SklearnScaler()
    X_tr   = scaler.fit_transform(X_tr)
    X_te   = scaler.transform(X_te)

    model = SklearnLR(
        solver="lbfgs",
        C=1.0, max_iter=1000, random_state=SEED,
    )
    model.fit(X_tr, y_tr)

    y_pred_tr = model.predict(X_tr)
    y_pred_te = model.predict(X_te)

    return {
        "name":       "scikit-learn\n(L-BFGS)",
        "losses":     None,
        "train_acc":  sk_accuracy(y_tr, y_pred_tr),
        "test_acc":   sk_accuracy(y_te, y_pred_te),
        "cm":         sk_confusion_matrix(y_te, y_pred_te),
        "macro_f1":   f1_score(y_te, y_pred_te, average="macro"),
        "macro_prec": precision_score(y_te, y_pred_te, average="macro"),
        "macro_rec":  recall_score(y_te, y_pred_te, average="macro"),
        "y_te":       y_te,
        "y_pred":     y_pred_te,
    }


# ── numpy metric helpers ────────────────────────────────────────────────────────

def _per_class_metric(y_true, y_pred, metric):
    classes = np.unique(y_true)
    vals = []
    cm = np_confusion_matrix(y_true, y_pred, n_classes=len(classes))
    for c in classes:
        tp = cm[c, c]
        fp = cm[:, c].sum() - tp
        fn = cm[c, :].sum() - tp
        p  = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        r  = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0.0
        vals.append({"p": p, "r": r, "f1": f1}[metric])
    return float(np.mean(vals))

_macro_f1_numpy   = lambda t, p: _per_class_metric(t, p, "f1")
_macro_prec_numpy = lambda t, p: _per_class_metric(t, p, "p")
_macro_rec_numpy  = lambda t, p: _per_class_metric(t, p, "r")


# ── Comparison table ───────────────────────────────────────────────────────────

def print_comparison(np_res, sk_res):
    W = 62
    print("\n" + "━" * W)
    print(f"{'COMPARISON: NumPy GD  vs.  scikit-learn':^{W}}")
    print("━" * W)
    header = f"{'Metric':<22} {'NumPy (GD)':>16} {'scikit-learn':>18}"
    print(header)
    print("─" * W)

    rows = [
        ("Test accuracy",    f"{np_res['test_acc']*100:>14.2f} %",  f"{sk_res['test_acc']*100:>16.2f} %"),
        ("Train accuracy",   f"{np_res['train_acc']*100:>14.2f} %", f"{sk_res['train_acc']*100:>16.2f} %"),
        ("Macro precision",  f"{np_res['macro_prec']:>15.4f}",       f"{sk_res['macro_prec']:>17.4f}"),
        ("Macro recall",     f"{np_res['macro_rec']:>15.4f}",        f"{sk_res['macro_rec']:>17.4f}"),
        ("Macro F1-score",   f"{np_res['macro_f1']:>15.4f}",         f"{sk_res['macro_f1']:>17.4f}"),
    ]
    for label, nv, sv in rows:
        print(f"  {label:<20} {nv} {sv}")

    diff = (sk_res["test_acc"] - np_res["test_acc"]) * 100
    print("─" * W)
    print(f"  Test accuracy gap : {diff:+.2f} percentage points (sklearn − NumPy)")
    print("━" * W + "\n")


# ── 4-panel figure ─────────────────────────────────────────────────────────────

def _draw_confusion_matrix(ax, cm, title, cmap):
    im = ax.imshow(cm, cmap=cmap, vmin=0)
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    ax.set_xticks(range(3))
    ax.set_yticks(range(3))
    ax.set_xticklabels(CLASS_NAMES, rotation=25, ha="right", fontsize=8)
    ax.set_yticklabels(CLASS_NAMES, fontsize=8)
    ax.set_xlabel("Predicted", fontsize=9)
    ax.set_ylabel("True", fontsize=9)
    ax.set_title(title, fontsize=10)
    thresh = cm.max() / 2
    for i in range(3):
        for j in range(3):
            ax.text(
                j, i, str(cm[i, j]),
                ha="center", va="center", fontsize=12, fontweight="bold",
                color="white" if cm[i, j] > thresh else "black",
            )


def save_comparison_figure(np_res, sk_res, path):
    fig = plt.figure(figsize=(13, 9))
    fig.suptitle(
        "Iris Dataset — NumPy Softmax Regression vs. scikit-learn",
        fontsize=14, fontweight="bold", y=0.98,
    )
    gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.42, wspace=0.38)

    # Panel 1 — loss curve
    ax1 = fig.add_subplot(gs[0, 0])
    iters = np.arange(1, len(np_res["losses"]) + 1)
    ax1.plot(iters, np_res["losses"], color="#2563EB", linewidth=1.5)
    ax1.set_xlabel("Iteration", fontsize=9)
    ax1.set_ylabel("Cross-entropy loss", fontsize=9)
    ax1.set_title("NumPy: Training Loss Curve", fontsize=10)
    ax1.grid(True, alpha=0.3)
    ax1.annotate(
        f"Final: {np_res['losses'][-1]:.4f}",
        xy=(iters[-1], np_res["losses"][-1]),
        xytext=(-80, 15), textcoords="offset points",
        fontsize=8, color="#2563EB",
        arrowprops=dict(arrowstyle="->", color="#2563EB", lw=0.8),
    )

    # Panel 2 — accuracy bar chart
    ax2 = fig.add_subplot(gs[0, 1])
    labels = ["Train", "Test"]
    np_vals = [np_res["train_acc"] * 100, np_res["test_acc"] * 100]
    sk_vals = [sk_res["train_acc"] * 100, sk_res["test_acc"] * 100]
    x = np.arange(len(labels))
    w = 0.32
    b1 = ax2.bar(x - w / 2, np_vals, w, label="NumPy GD",    color="#2563EB", alpha=0.85)
    b2 = ax2.bar(x + w / 2, sk_vals, w, label="scikit-learn", color="#16A34A", alpha=0.85)
    ax2.set_ylim(90, 102)
    ax2.set_ylabel("Accuracy (%)", fontsize=9)
    ax2.set_title("Accuracy Comparison", fontsize=10)
    ax2.set_xticks(x)
    ax2.set_xticklabels(labels, fontsize=9)
    ax2.legend(fontsize=8)
    ax2.grid(axis="y", alpha=0.3)
    for bar_group in [b1, b2]:
        for bar in bar_group:
            ax2.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.15,
                f"{bar.get_height():.1f}",
                ha="center", va="bottom", fontsize=7.5,
            )

    # Panel 3 — NumPy confusion matrix
    ax3 = fig.add_subplot(gs[1, 0])
    _draw_confusion_matrix(
        ax3, np_res["cm"],
        f"NumPy Confusion Matrix\nTest acc = {np_res['test_acc']*100:.2f} %",
        "Blues",
    )

    # Panel 4 — sklearn confusion matrix
    ax4 = fig.add_subplot(gs[1, 1])
    _draw_confusion_matrix(
        ax4, sk_res["cm"],
        f"sklearn Confusion Matrix\nTest acc = {sk_res['test_acc']*100:.2f} %",
        "Greens",
    )

    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Comparison figure saved → {path}")


# ── Main ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    iris   = load_iris()
    X, y   = iris.data.astype(np.float64), iris.target

    print("Running NumPy pipeline …")
    np_res = run_numpy(X, y)

    print("Running scikit-learn pipeline …")
    sk_res = run_sklearn(X, y)

    print_comparison(np_res, sk_res)

    fig_path = os.path.join(PLOT_DIR, "comparison.png")
    save_comparison_figure(np_res, sk_res, fig_path)
