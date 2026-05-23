"""
numpy_lr.model
==============
Softmax (multinomial logistic) regression trained with full-batch gradient
descent — implemented using NumPy only.

Theory recap
------------
Given an input matrix X ∈ ℝⁿˣᵈ, weight matrix W ∈ ℝᵈˣᴷ, and bias b ∈ ℝ¹ˣᴷ:

  Forward pass
  ────────────
  Z  = X W + b                        linear scores    (n, K)
  P  = softmax(Z)                     class probs      (n, K)
       where softmax(z)_k = exp(z_k) / Σ_j exp(z_j)

  Loss (cross-entropy)
  ────────────────────
  L  = −(1/n) Σᵢ Σₖ Yᵢₖ log Pᵢₖ
  where Y is the one-hot encoding of the true labels.

  Backward pass (chain rule)
  ──────────────────────────
  ∂L/∂Z = (1/n)(P − Y)               error signal     (n, K)
  ∂L/∂W = Xᵀ ∂L/∂Z                  weight gradient  (d, K)
  ∂L/∂b = Σᵢ (∂L/∂Z)ᵢ              bias gradient    (1, K)

  Update
  ──────
  W ← W − η ∂L/∂W
  b ← b − η ∂L/∂b
"""

import numpy as np


class SoftmaxRegression:
    """
    Multiclass logistic regression via softmax + cross-entropy + batch GD.

    Parameters
    ----------
    learning_rate : float
        Step size η for gradient descent.  Larger values converge faster
        but may overshoot; smaller values are stable but slow.
    n_iters : int
        Total number of gradient descent updates.
    random_state : int | None
        Seed for reproducible weight initialisation.

    Attributes set after fit()
    --------------------------
    W      : ndarray (n_features, n_classes)   — learned weights
    b      : ndarray (1, n_classes)            — learned bias
    losses : list[float]                       — cross-entropy per iteration
    """

    def __init__(
        self,
        learning_rate: float = 0.5,
        n_iters: int = 2000,
        random_state: int | None = 42,
    ) -> None:
        self.learning_rate = learning_rate
        self.n_iters       = n_iters
        self.random_state  = random_state
        self.W:      np.ndarray | None = None
        self.b:      np.ndarray | None = None
        self.losses: list[float]       = []

    # ── private helpers ────────────────────────────────────────────────────

    @staticmethod
    def _softmax(Z: np.ndarray) -> np.ndarray:
        """
        Numerically stable softmax: subtract the row-maximum before exp so
        the largest value becomes exp(0) = 1, preventing overflow.
        The result is mathematically identical to the standard definition.
        """
        shifted = Z - Z.max(axis=1, keepdims=True)
        exp_Z   = np.exp(shifted)
        return exp_Z / exp_Z.sum(axis=1, keepdims=True)

    @staticmethod
    def _one_hot(y: np.ndarray, n_classes: int) -> np.ndarray:
        """Convert integer labels to a one-hot matrix Y ∈ {0,1}^(n×K)."""
        n  = len(y)
        oh = np.zeros((n, n_classes))
        oh[np.arange(n), y] = 1.0
        return oh

    @staticmethod
    def _cross_entropy(P: np.ndarray, Y_oh: np.ndarray) -> float:
        """
        Mean cross-entropy: L = −mean(Σ_k Y_k log P_k).
        np.clip prevents log(0) = −∞ from polluting the loss.
        """
        return -np.mean(np.sum(Y_oh * np.log(np.clip(P, 1e-12, 1.0)), axis=1))

    # ── public API ─────────────────────────────────────────────────────────

    def fit(self, X: np.ndarray, y: np.ndarray) -> "SoftmaxRegression":
        """
        Fit weights W and bias b using batch gradient descent.

        Parameters
        ----------
        X : (n_samples, n_features) — already normalised feature matrix
        y : (n_samples,)            — integer class labels 0 … K-1
        """
        rng     = np.random.RandomState(self.random_state)
        n, d    = X.shape
        k       = len(np.unique(y))

        # Small random init breaks the symmetry that would keep all class
        # weights identical throughout training.
        self.W      = rng.randn(d, k) * 0.01
        self.b      = np.zeros((1, k))
        Y_oh        = self._one_hot(y, k)
        self.losses = []

        for _ in range(self.n_iters):
            # ── Forward pass ─────────────────────────────────────────────
            Z    = X @ self.W + self.b      # (n, k) — linear scores
            P    = self._softmax(Z)          # (n, k) — class probabilities
            loss = self._cross_entropy(P, Y_oh)
            self.losses.append(loss)

            # ── Backward pass ────────────────────────────────────────────
            dZ = (P - Y_oh) / n             # (n, k) — scaled error signal
            dW = X.T @ dZ                   # (d, k) — weight gradient
            db = dZ.sum(axis=0, keepdims=True)  # (1, k) — bias gradient

            # ── Gradient descent update ──────────────────────────────────
            self.W -= self.learning_rate * dW
            self.b -= self.learning_rate * db

        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Return the softmax probability matrix P ∈ [0,1]^(n×K)."""
        if self.W is None:
            raise RuntimeError("Call fit() before predict_proba().")
        return self._softmax(X @ self.W + self.b)

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Return the predicted class index (argmax of probabilities)."""
        return np.argmax(self.predict_proba(X), axis=1)
