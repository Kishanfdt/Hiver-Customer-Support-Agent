"""
src/escalation.py - Pure-Python deterministic escalation gate.

Evaluates intent classification confidence, retrieval similarity, and sensitive
policy categories to route each interaction to human agents or automated handling.
"""

from typing import Dict
from src import config


def decide(intent: str, confidence: float, top_similarity: float) -> Dict[str, str]:
    """
    Decides whether to escalate to human agent or auto-handle.

    Rules:
    1. Always-escalate intents (e.g. billing_dispute, account_access) -> escalate.
    2. Intent confidence below CONFIDENCE_ESCALATION_THRESHOLD -> escalate.
    3. Retrieval similarity below RETRIEVAL_SIMILARITY_FLOOR -> escalate.
    4. Otherwise -> auto_handle.

    Returns:
        dict: {"decision": "escalate" | "auto_handle", "reason": "<explanation>"}
    """
    # 1. Check sensitive policy category
    if intent in config.ALWAYS_ESCALATE_INTENTS:
        return {
            "decision": "escalate",
            "reason": f"Always-escalate intent policy applied for sensitive category: '{intent}'"
        }

    # 2. Check classifier confidence
    if confidence < config.CONFIDENCE_ESCALATION_THRESHOLD:
        return {
            "decision": "escalate",
            "reason": (
                f"Classification confidence ({confidence:.2f}) is below threshold "
                f"({config.CONFIDENCE_ESCALATION_THRESHOLD})"
            )
        }

    # 3. Check retrieval similarity floor
    if top_similarity < config.RETRIEVAL_SIMILARITY_FLOOR:
        return {
            "decision": "escalate",
            "reason": (
                f"Precedent retrieval similarity ({top_similarity:.2f}) is below floor "
                f"({config.RETRIEVAL_SIMILARITY_FLOOR})"
            )
        }

    # 4. Qualified for automated resolution
    return {
        "decision": "auto_handle",
        "reason": (
            f"Sufficient confidence ({confidence:.2f} >= {config.CONFIDENCE_ESCALATION_THRESHOLD}) "
            f"and precedent match ({top_similarity:.2f} >= {config.RETRIEVAL_SIMILARITY_FLOOR})"
        )
    }
