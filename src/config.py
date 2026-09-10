"""
src/config.py - Centralized configuration and single tunable source for the support agent.

All adjustable hyperparameters, intent schemas, escalation thresholds, model identifiers,
and retrieval settings are defined here. Changes to agent behavior, confidence thresholds,
and taxonomy should be made here rather than scattered across modules.
"""

from typing import List, Set

# Target brand for customer support interactions
BRAND: str = "AmazonHelp"

# Intent taxonomy derived empirically from customer tweet clustering
INTENTS: List[str] = [
    "order_status",      # Where is my order, tracking requests, shipment updates
    "refund_request",    # Inquiries about refund processing, return status, money back
    "delivery_issue",    # Package delayed, marked delivered but missing, damaged delivery
    "account_access",    # Password resets, account locked, 2FA issues, login trouble
    "billing_dispute",   # Unexpected charges, double billing, payment failure, Prime fees
    "product_defect",    # Broken item, wrong item sent, defective or counterfeit product
    "general_inquiry"    # Policy questions, hours, general store queries, feedback
]

# Sensitive intents that must always be routed to human agents for security/compliance
ALWAYS_ESCALATE_INTENTS: Set[str] = {
    "billing_dispute",
    "account_access"
}

# Thresholds for autonomous resolution gate
CONFIDENCE_ESCALATION_THRESHOLD: float = 0.55  # Minimum intent confidence to auto-handle
RETRIEVAL_SIMILARITY_FLOOR: float = 0.15       # Minimum cosine similarity to retrieved precedents
TOP_K_RETRIEVAL: int = 3                       # Number of resolution precedents to retrieve

# Gemini LLM models
CLASSIFIER_MODEL: str = "gemini-3.6-flash"
GENERATION_MODEL: str = "gemini-3.6-flash"
JUDGE_MODEL: str = "gemini-3.6-flash"

# Character limit for public social media replies
MAX_REPLY_CHARS: int = 280
