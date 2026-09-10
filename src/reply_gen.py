"""
src/reply_gen.py - Grounded reply generation module matching brand tone and policies.

Generates Twitter responses (< 280 characters) strictly adhering to retrieved
precedents without hallucinating financial figures, promises, or arbitrary timelines.
"""

from typing import Any, Dict, List, Optional
from src import config
from src.llm_client import LLMClient


FALLBACK_REPLIES = {
    "order_status": "We'd like to look into your delivery status. Please send your 17-digit order number via DM so we can check the latest tracking details for you.",
    "refund_request": "We want to help with your return and refund. Please reach out to us via direct message with your order details so our team can assist.",
    "delivery_issue": "We apologize for the delivery trouble. Please DM us your order ID and confirmed delivery address so we can investigate with the carrier.",
    "account_access": "For your security, we cannot discuss account credentials publicly. Please visit amazon.com/help or contact our secure support team directly.",
    "billing_dispute": "We take billing concerns seriously. Because payment details are sensitive, please contact our billing department directly through your Amazon account.",
    "product_defect": "We're sorry your item arrived in that condition. Please send us a DM with your order number so we can arrange a replacement or return.",
    "general_inquiry": "We're here to help! Please reach out to us via direct message with more details about your inquiry so we can look into it."
}


class ReplyGenerator:
    """
    Generates brand-aligned replies grounded in retrieved precedents.
    """

    SYSTEM_PROMPT = """You are the official Twitter customer care voice for AmazonHelp.
Your task is to draft a helpful, polite, and concise reply to the customer's tweet.

STRICT REQUIREMENTS:
1. Under 280 characters total (Twitter limit).
2. Ground your response in the provided past precedents and typical Amazon customer service procedures.
3. ANTI-HALLUCINATION: Do NOT invent specific dollar refund amounts, specific dates/timelines, or unverified policy promises not grounded in the precedents.
4. For security, never ask for passwords or full card numbers. Direct sensitive issues to secure DM or official help links.
5. Return JSON: {"reply": "<draft_reply_text>"}
"""

    def __init__(self, llm_client: LLMClient):
        self.client = llm_client

    def generate(
        self,
        message: str,
        intent: str,
        precedents: Optional[List[Dict[str, Any]]] = None
    ) -> str:
        """
        Drafts a grounded reply for the customer query.
        """
        precedents = precedents or []

        # Construct context from top precedents
        precedents_context = ""
        for i, p in enumerate(precedents[:config.TOP_K_RETRIEVAL], 1):
            precedents_context += (
                f"\nPrecedent {i} (Similarity {p.get('similarity', 0.0):.2f}):\n"
                f"Customer: \"{p.get('text', '')}\"\n"
                f"Brand Resolution: \"{p.get('resolution_text', '')}\"\n"
            )

        user_prompt = f"""Customer Intent: {intent}
Customer Tweet: "{message}"

Retrieved Precedents:
{precedents_context if precedents_context else "None available."}

Draft an official response under 280 characters:
"""

        result = self.client.complete_json(self.SYSTEM_PROMPT, user_prompt, max_tokens=150)

        # Fallback if offline or parsing failure
        if result.get("_offline_stub") or "reply" not in result:
            return self._fallback_reply(intent, precedents)

        reply = result.get("reply", "").strip()

        # Enforce character limit
        if len(reply) > config.MAX_REPLY_CHARS:
            reply = reply[:config.MAX_REPLY_CHARS - 3].rstrip() + "..."

        return reply or self._fallback_reply(intent, precedents)

    def _fallback_reply(self, intent: str, precedents: List[Dict[str, Any]]) -> str:
        """Deterministic offline fallback grounded in top precedent or intent template."""
        if precedents and precedents[0].get("resolution_text"):
            # Return top precedent resolution if valid length, otherwise intent template
            prec_reply = precedents[0]["resolution_text"].strip()
            if len(prec_reply) <= config.MAX_REPLY_CHARS:
                return prec_reply

        return FALLBACK_REPLIES.get(intent, FALLBACK_REPLIES["general_inquiry"])
