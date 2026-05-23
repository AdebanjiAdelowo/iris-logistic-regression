"""
numpy_lr.preprocessing
======================
Data preparation utilities implemented with NumPy only — no scikit-learn.

Classes
-------
StandardScaler
    Subtracts the column mean and divides by the column standard deviation
    so that every feature has mean ≈ 0 and std ≈ 1.  This is critical for
    gradient descent: un-normalised features with very different scales cause
    the loss landscape to be elongated in some directions, making it hard to
    choose a single learning rate that works for all weights.

Functions
---------
train_test_split
    Randomly permutes the dataset and carves off a held-out test portion.
"""

import numpy as np


class StandardScaler:
    """
    Standardise features by removing the mean and scaling to unit variance.

    Attributes set after fit()
    --------------------------
    mean_ : ndarray of shape (n_features,)
    scale_ : ndarray of shape (n_features,)  — standard deviation per column

    Notes
    -----
    Constant columns (std == 0) are left unchanged to avoid division by zero.
    The scaler must be fitted on training data only; calling transform() on the
    test set uses the *training* statistics to prevent data leakage.
    """

    def fit(self, X: np.ndarray) -> "StandardScaler":
        self.mean_  = X.mean(axis=0)
        self.scale_ = X.std(axis=0)
        self.scale_[self.scale_ == 0] = 1.0   # guard constant columns
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        return (X - self.mean_) / self.scale_

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        return self.fit(X).transform(X)


def train_test_split(
    X: np.ndarray,
    y: np.ndarray,
    *,
    test_size: float = 0.2,
    random_state: int | None = 42,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Randomly shuffle and split arrays into train / test subsets.

    Parameters
    ----------
    X            : feature matrix  (n_samples, n_features)
    y            : label vector    (n_samples,)
    test_size    : fraction of samples to reserve for testing
    random_state : seed for the NumPy random number generator

    Returns
    -------
    X_train, X_test, y_train, y_test
    """
    rng = np.random.RandomState(random_state)
    n   = len(y)
    idx = rng.permutation(n)

    n_test  = max(1, int(np.ceil(n * test_size)))
    test_idx  = idx[:n_test]
    train_idx = idx[n_test:]

    return X[train_idx], X[test_idx], y[train_idx], y[test_idx]
