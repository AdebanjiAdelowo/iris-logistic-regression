"""
numpy_lr — softmax regression built entirely with NumPy.

Exports
-------
SoftmaxRegression  : gradient-descent multiclass classifier
StandardScaler     : zero-mean / unit-variance feature normaliser
train_test_split   : reproducible stratified-style data split
accuracy_score     : fraction of correct predictions
confusion_matrix   : K×K count matrix
classification_report : per-class precision / recall / F1
"""

from .model import SoftmaxRegression
from .preprocessing import StandardScaler, train_test_split
from .metrics import accuracy_score, confusion_matrix, classification_report

__all__ = [
    "SoftmaxRegression",
    "StandardScaler",
    "train_test_split",
    "accuracy_score",
    "confusion_matrix",
    "classification_report",
]
