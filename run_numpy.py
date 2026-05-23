"""
run_numpy.py — Full Iris classification pipeline using NumPy only.

Steps
-----
1. Load the Iris dataset via scikit-learn (data loading only; no sklearn
   is used for modelling, preprocessing, or evaluation).
2. Normalise features with a hand-rolled StandardScaler.
3. Split into 80 % train / 20 % test with a custom train_test_split.
4. Train a SoftmaxRegression model with batch gradient descent.
5. Evaluate: accuracy + classification report + confusion matrix.
6. Save the training loss curve and confusion matrix to plots/.

Run
---
    python run_numpy.py
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from sklearn.datasets import load_iris   # data loading only

from numpy_lr import (
    SoftmaxRegression,
    StandardScaler,
    train_test_split,
    accuracy_score,
    confusion_matrix,
    classification_report,
)

SEED       = 42
PLOT_DIR   = "plots"
CLASS_NAMES = ["Iris-setosa", "Iris-versicolor", "Iris-virginica"]

os.makedirs(PLOT_DIR, exist_ok=True)


# ── 1. Load data ───────────────────────────────────────────────────────────────

iris   = load_iris()
X, y   = iris.data.astype(np.float64), iris.target
n, d   = X.shape

print("=" * 60)
print("  NumPy-only Softmax Regression  ·  Iris Dataset")
print("=" * 60)
print(f"  Samples : {n}  |  Features : {d}  |  Classes : {len(np.unique(y))}")
print(f"  Feature names : {iris.feature_names}")
print()


# ── 2. Preprocess ──────────────────────────────────────────────────────────────

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=SEED
)

scaler  = StandardScaler()
X_train = scaler.fit_transform(X_train)   # fit on train, apply to train
X_test  = scaler.transform(X_test)        # apply training stats to test (no leakage)

print(f"  Train : {len(y_train)} samples  |  Test : {len(y_test)} samples")
print(f"  Feature mean after scaling : {X_train.mean(axis=0).round(3)}")
print(f"  Feature std  after scaling : {X_train.std(axis=0).round(3)}")
print()


# ── 3. Train ───────────────────────────────────────────────────────────────────

model = SoftmaxRegression(learning_rate=0.5, n_iters=2000, random_state=SEED)

print("  Training …")
model.fit(X_train, y_train)

print(f"  Initial loss : {model.losses[0]:.6f}")
print(f"  Final   loss : {model.losses[-1]:.6f}")
print(f"  Reduction    : {(1 - model.losses[-1] / model.losses[0]) * 100:.1f} %")
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
print()

cm = confusion_matrix(y_test, y_pred_test, n_classes=3)
print("  Confusion matrix (test set)  — rows: true, cols: predicted")
print("  " + "  ".join(f"{n:>14}" for n in CLASS_NAMES))
for i, row in enumerate(cm):
    bar = "  ".join(f"{v:>14}" for v in row)
    print(f"  {CLASS_NAMES[i]:<16}  {bar}")
print()


# ── 5. Plot: training loss curve ───────────────────────────────────────────────

fig, ax = plt.subplots(figsize=(8, 4))
iters   = np.arange(1, len(model.losses) + 1)
ax.plot(iters, model.losses, color="#2563EB", linewidth=1.5, label="Training loss")
ax.set_xlabel("Iteration", fontsize=12)
ax.set_ylabel("Cross-entropy loss", fontsize=12)
ax.set_title(
    f"NumPy Softmax Regression — Loss Curve\n"
    f"lr={model.learning_rate}, iters={model.n_iters}, "
    f"final loss={model.losses[-1]:.4f}",
    fontsize=11,
)
ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{int(x):,}"))
ax.legend(fontsize=11)
ax.grid(True, alpha=0.3)
fig.tight_layout()
loss_path = os.path.join(PLOT_DIR, "numpy_loss_curve.png")
fig.savefig(loss_path, dpi=150)
plt.close(fig)
print(f"  Loss curve saved → {loss_path}")


# ── 6. Plot: confusion matrix ──────────────────────────────────────────────────

fig, ax = plt.subplots(figsize=(6, 5))
im = ax.imshow(cm, cmap="Blues", vmin=0)
plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
ax.set_xticks(range(3))
ax.set_yticks(range(3))
ax.set_xticklabels(CLASS_NAMES, rotation=30, ha="right", fontsize=9)
ax.set_yticklabels(CLASS_NAMES, fontsize=9)
ax.set_xlabel("Predicted label", fontsize=11)
ax.set_ylabel("True label", fontsize=11)
ax.set_title(
    f"NumPy Confusion Matrix (test set)\nAccuracy = {test_acc * 100:.2f} %",
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
cm_path = os.path.join(PLOT_DIR, "numpy_confusion_matrix.png")
fig.savefig(cm_path, dpi=150)
plt.close(fig)
print(f"  Confusion matrix saved → {cm_path}")

print()
print("=" * 60)
