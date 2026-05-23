"""
run_sklearn.py — Iris classification pipeline using scikit-learn.

Uses the identical preprocessing strategy (StandardScaler + 80/20 split)
as run_numpy.py so the two results can be compared fairly.

Steps
-----
1. Load Iris.
2. Normalise with sklearn's StandardScaler.
3. Split with sklearn's train_test_split (same random_state=42).
4. Train sklearn LogisticRegression (L-BFGS solver, L2 regularisation).
5. Evaluate and save confusion matrix to plots/.

Run
---
    python run_sklearn.py
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    classification_report,
)

SEED        = 42
PLOT_DIR    = "plots"
CLASS_NAMES = ["Iris-setosa", "Iris-versicolor", "Iris-virginica"]

os.makedirs(PLOT_DIR, exist_ok=True)


# ── 1. Load data ───────────────────────────────────────────────────────────────

iris = load_iris()
X, y = iris.data, iris.target

print("=" * 60)
print("  scikit-learn Logistic Regression  ·  Iris Dataset")
print("=" * 60)
print(f"  Samples : {len(y)}  |  Features : {X.shape[1]}  |  Classes : 3")
print()


# ── 2. Preprocess ──────────────────────────────────────────────────────────────

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=SEED, stratify=y
)

scaler  = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test  = scaler.transform(X_test)

print(f"  Train : {len(y_train)} samples  |  Test : {len(y_test)} samples")
print()


# ── 3. Train ───────────────────────────────────────────────────────────────────

# LogisticRegression uses L-BFGS by default for multi_class='multinomial'.
# C=1.0 is the inverse regularisation strength (matches sklearn default).
# max_iter=1000 ensures convergence.
model = LogisticRegression(
    solver="lbfgs",
    C=1.0,
    max_iter=1000,
    random_state=SEED,
)
model.fit(X_train, y_train)

print("  Model parameters:")
print(f"    solver     : {model.solver}")
print(f"    C (inv λ)  : {model.C}")
print(f"    iterations : {model.n_iter_[0]}")
print(f"    W shape    : {model.coef_.shape}")
print()


# ── 4. Evaluate ────────────────────────────────────────────────────────────────

y_pred_train = model.predict(X_train)
y_pred_test  = model.predict(X_test)

train_acc = accuracy_score(y_train, y_pred_train)
test_acc  = accuracy_score(y_test,  y_pred_test)

print(f"  Train accuracy : {train_acc * 100:.2f} %")
print(f"  Test  accuracy : {test_acc  * 100:.2f} %")
print()

print("  Classification report (test set):")
print(classification_report(y_test, y_pred_test, target_names=CLASS_NAMES))

cm = confusion_matrix(y_test, y_pred_test)
print("  Confusion matrix (test set)  — rows: true, cols: predicted")
print("  " + "  ".join(f"{n:>14}" for n in CLASS_NAMES))
for i, row in enumerate(cm):
    bar = "  ".join(f"{v:>14}" for v in row)
    print(f"  {CLASS_NAMES[i]:<16}  {bar}")
print()


# ── 5. Plot: confusion matrix ──────────────────────────────────────────────────

fig, ax = plt.subplots(figsize=(6, 5))
im = ax.imshow(cm, cmap="Greens", vmin=0)
plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
ax.set_xticks(range(3))
ax.set_yticks(range(3))
ax.set_xticklabels(CLASS_NAMES, rotation=30, ha="right", fontsize=9)
ax.set_yticklabels(CLASS_NAMES, fontsize=9)
ax.set_xlabel("Predicted label", fontsize=11)
ax.set_ylabel("True label", fontsize=11)
ax.set_title(
    f"sklearn Confusion Matrix (test set)\nAccuracy = {test_acc * 100:.2f} %",
    fontsize=11,
)
thresh = cm.max() / 2
for i in range(3):
    for j in range(3):
        ax.text(
            j, i, str(cm[i, j]),
            ha="center", va="center", fontsize=13, fontweight="bold",
            color="white" if cm[i, j] > thresh else "black",
        )
fig.tight_layout()
cm_path = os.path.join(PLOT_DIR, "sklearn_confusion_matrix.png")
fig.savefig(cm_path, dpi=150)
plt.close(fig)
print(f"  Confusion matrix saved → {cm_path}")

print()
print("=" * 60)
