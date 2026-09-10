"""
src/classifier.py - Intent classification module using Gemini LLM with keyword fallback.

Classifies incoming customer tweets into one of the 7 empirical intent categories.
"""

from typing import Any, Dict, List, Tuple
from src import config
from src.llm_client import LLMClient


# Keyword mapping used for offline fallback and simple baseline
KEYWORD_RULES: List[Tuple[str, List[str]]] = [
    ("billing_dispute", [
        "charged", "charge", "overcharged", "double charged", "unauthorized",
        "card", "bank", "billing", "deducted", "prime fee", "subscription fee", "statement"
    ]),
    ("account_access", [
        "password", "log in", "login", "locked", "otp", "2fa", "sign in", "signin",
        "verification code", "hacked", "reset password", "can't access", "account blocked"
    ]),
    ("refund_request", [
        "refund", "money back", "reimburse", "reimbursement", "return", "returned",
        "cancel order", "cancellation", "credited back"
    ]),
    ("delivery_issue", [
        "delivery", "delivered", "driver", "carrier", "ups", "usps", "fedex",
        "late delivery", "delayed", "marked delivered", "missing package", "never arrived"
    ]),
    ("order_status", [
        "where is my order", "track", "tracking", "status", "dispatch", "dispatched",
        "shipped", "hasn't shipped", "not shipped", "order number", "eta"
    ]),
    ("product_defect", [
        "broken", "damaged", "defective", "faulty", "wrong item", "wrong product",
        "different item", "fake", "counterfeit", "not working", "scratched", "leaking"
    ]),
    ("general_inquiry", [
        "question", "help", "info", "information", "policy", "hours", "contact",
        "service", "customer service", "agent"
    ])
]


def rule_based_classify(text: str) -> Dict[str, Any]:
    """
    Deterministic rule-based keyword classifier.
    Serves as offline fallback and Simple Baseline.
    """
    text_lower = text.lower()

    best_intent = "general_inquiry"
    max_matches = 0

    for intent, keywords in KEYWORD_RULES:
        matches = sum(1 for kw in keywords if kw in text_lower)
        if matches > max_matches:
            max_matches = matches
            best_intent = intent

    if max_matches > 0:
        confidence = min(0.60 + 0.15 * max_matches, 0.95)
    else:
        # Default fallback
        best_intent = "general_inquiry"
        confidence = 0.50

    return {
        "intent": best_intent,
        "confidence": round(confidence, 2)
    }


class IntentClassifier:
    """
    Intent Classifier using Gemini LLM with few-shot prompting,
    falling back seamlessly to rule_based_classify when offline.
    """

    SYSTEM_PROMPT = f"""You are an expert customer service triage agent for AmazonHelp on Twitter.
Analyze the user's message and categorize it into EXACTLY ONE of the following 7 intents:
- order_status: Inquiries about tracking, shipment progress, order status, or when an item will ship.
- refund_request: Requests for refund status, money back, returning an item, or cancellation refunds.
- delivery_issue: Delivery delays, packages marked delivered but not received, carrier/driver problems.
- account_access: Login troubles, password reset, 2FA/OTP issues, locked or compromised accounts.
- billing_dispute: Unexpected card charges, double charges, unauthorized charges, Prime membership billing.
- product_defect: Received broken, damaged, defective, expired, or incorrect/wrong items.
- general_inquiry: General service inquiries, business policies, store hours, or general questions.

Respond with ONLY a JSON object:
{{
  "intent": "<one_of_the_above_7_intents>",
  "confidence": <float between 0.0 and 1.0>
}}
"""

    FEW_SHOT_EXAMPLES = """
Example 1:
User: "Where is my package? It was supposed to arrive yesterday and tracking hasn't updated."
Response: {"intent": "order_status", "confidence": 0.95}

Example 2:
User: "I returned the shoes two weeks ago but still have not received my money back!"
Response: {"intent": "refund_request", "confidence": 0.98}

Example 3:
User: "Why was my credit card charged $14.99 twice this morning? I did not authorize this!"
Response: {"intent": "billing_dispute", "confidence": 0.96}

Example 4:
User: "I am locked out of my Amazon account and not receiving the SMS verification code."
Response: {"intent": "account_access", "confidence": 0.97}

Example 5:
User: "The blender arrived shattered inside the box. Glass everywhere."
Response: {"intent": "product_defect", "confidence": 0.95}
"""

    def __init__(self, llm_client: LLMClient):
        self.client = llm_client

    def classify(self, text: str) -> Dict[str, Any]:
        """
        Classifies input tweet text. Returns dict with 'intent' and 'confidence'.
        """
        if not text or not text.strip():
            return {"intent": "general_inquiry", "confidence": 0.50}

        prompt = f"{self.FEW_SHOT_EXAMPLES}\nNow classify this customer tweet:\nUser: \"{text}\"\nResponse:"
        result = self.client.complete_json(self.SYSTEM_PROMPT, prompt, max_tokens=150)

        # If offline stub or error, fallback to keyword rules
        if result.get("_offline_stub") or "intent" not in result:
            return rule_based_classify(text)

        intent = result.get("intent", "").strip().lower()
        if intent not in config.INTENTS:
            # Reconcile or fallback if model hallucinates intent outside config
            for valid_intent in config.INTENTS:
                if valid_intent in intent or intent in valid_intent:
                    intent = valid_intent
                    break
            else:
                return rule_based_classify(text)

        try:
            confidence = float(result.get("confidence", 0.85))
            confidence = max(0.0, min(1.0, confidence))
        except (ValueError, TypeError):
            confidence = 0.80

        return {
            "intent": intent,
            "confidence": round(confidence, 2)
        }
