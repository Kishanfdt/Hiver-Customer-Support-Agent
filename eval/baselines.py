"""
eval/baselines.py - Baseline models for comparative evaluation.

1. Trivial Baseline: Always predicts the majority-class intent ("order_status")
   and never escalates ("auto_handle").
2. Simple Baseline: Rule-based keyword classifier with static intent-to-decision mapping.
"""

from typing import Any, Dict
from src import config
from src.classifier import rule_based_classify

# Majority intent identified from empirical exploration
MAJORITY_INTENT: str = "order_status"


def trivial_baseline(text: str) -> Dict[str, Any]:
    """
    Trivial baseline:
    - Intent: Always predicts majority class ('order_status')
    - Confidence: 1.0
    - Escalation Decision: Always 'auto_handle' (never escalates)
    """
    return {
        "predicted_intent": MAJORITY_INTENT,
        "confidence": 1.0,
        "decision": "auto_handle",
        "decision_reason": "Trivial baseline default: majority class, never escalate"
    }


def simple_baseline(text: str) -> Dict[str, Any]:
    """
    Simple baseline:
    - Intent: Keyword rule-based heuristic classifier
    - Escalation Decision: Static mapping (sensitive intents escalate, others auto-handle)
    """
    clf = rule_based_classify(text)
    intent = clf["intent"]
    confidence = clf["confidence"]

    # Fixed intent-to-decision mapping
    if intent in config.ALWAYS_ESCALATE_INTENTS:
        decision = "escalate"
        reason = f"Simple baseline rule: sensitive intent '{intent}' escalates"
    else:
        decision = "auto_handle"
        reason = f"Simple baseline rule: intent '{intent}' auto-handled"

    return {
        "predicted_intent": intent,
        "confidence": confidence,
        "decision": decision,
        "decision_reason": reason
    }
