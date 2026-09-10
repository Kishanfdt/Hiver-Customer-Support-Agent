"""
eval/metrics.py - Quantitative evaluation metrics for intent classification and escalation.
"""

from typing import Dict, List
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score


def intent_metrics(y_true: List[str], y_pred: List[str]) -> Dict[str, float]:
    """
    Computes intent classification accuracy and macro-averaged F1.
    """
    if not y_true or not y_pred or len(y_true) != len(y_pred):
        return {"accuracy": 0.0, "macro_f1": 0.0, "weighted_f1": 0.0}

    acc = accuracy_score(y_true, y_pred)
    macro_f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)
    weighted_f1 = f1_score(y_true, y_pred, average="weighted", zero_division=0)

    return {
        "accuracy": round(float(acc), 4),
        "macro_f1": round(float(macro_f1), 4),
        "weighted_f1": round(float(weighted_f1), 4)
    }


def escalation_metrics(y_true: List[str], y_pred: List[str]) -> Dict[str, float]:
    """
    Computes escalation gate metrics treating 'escalate' as the positive class.
    """
    if not y_true or not y_pred or len(y_true) != len(y_pred):
        return {
            "precision": 0.0,
            "recall": 0.0,
            "f1": 0.0,
            "accuracy": 0.0,
            "escalation_rate": 0.0
        }

    pos_label = "escalate"
    precision = precision_score(y_true, y_pred, pos_label=pos_label, zero_division=0)
    recall = recall_score(y_true, y_pred, pos_label=pos_label, zero_division=0)
    f1 = f1_score(y_true, y_pred, pos_label=pos_label, zero_division=0)
    acc = accuracy_score(y_true, y_pred)
    esc_rate = sum(1 for p in y_pred if p == pos_label) / len(y_pred) if y_pred else 0.0

    return {
        "precision": round(float(precision), 4),
        "recall": round(float(recall), 4),
        "f1": round(float(f1), 4),
        "accuracy": round(float(acc), 4),
        "escalation_rate": round(float(esc_rate), 4)
    }
