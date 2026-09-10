"""
tests/test_pipeline.py - Offline unit tests verifying data pairing, retrieval leakage protection,
and escalation decision branches without requiring API keys.
"""

import pandas as pd
import pytest
from src import config
from src.data_prep import build_brand_pairs
from src.escalation import decide
from src.retrieval import ResolutionRetriever


def test_build_brand_pairs():
    """Verify pairing of inbound customer tweets with brand reply tweets."""
    sample_df = pd.DataFrame([
        {
            "tweet_id": "1",
            "author_id": "cust_1",
            "inbound": True,
            "text": "Where is my package?",
            "in_response_to_tweet_id": ""
        },
        {
            "tweet_id": "2",
            "author_id": "AmazonHelp",
            "inbound": False,
            "text": "We can help you track that. Please DM us your order ID.",
            "in_response_to_tweet_id": "1"
        },
        {
            "tweet_id": "3",
            "author_id": "cust_2",
            "inbound": True,
            "text": "Unanswered question",
            "in_response_to_tweet_id": ""
        }
    ])

    pairs = build_brand_pairs(sample_df, brand="AmazonHelp")
    assert len(pairs) == 1
    assert pairs.iloc[0]["tweet_id"] == "1"
    assert pairs.iloc[0]["text"] == "Where is my package?"
    assert "Please DM us" in pairs.iloc[0]["resolution_text"]


def test_retrieval_top_k_and_leakage_exclusion():
    """Verify retrieval returns relevant precedents and strictly respects exclude_ids."""
    pairs_df = pd.DataFrame([
        {
            "tweet_id": "101",
            "text": "Where is my delayed delivery order?",
            "resolution_text": "Please DM us your tracking number."
        },
        {
            "tweet_id": "102",
            "text": "Refund not received for returned item",
            "resolution_text": "Refunds process within 3-5 business days."
        },
        {
            "tweet_id": "103",
            "text": "I was double charged on my credit card",
            "resolution_text": "Check billing at amazon.com/manage."
        }
    ])

    retriever = ResolutionRetriever()
    retriever.fit(pairs_df)

    # Query matching order delivery
    query = "My delivery is late and order is delayed"
    results = retriever.top_k(query, k=2)
    assert len(results) > 0
    assert results[0]["tweet_id"] == "101"
    assert results[0]["similarity"] > 0.10

    # Leakage Guard Test: Exclude tweet_id 101
    results_excluded = retriever.top_k(query, k=2, exclude_ids={"101"})
    retrieved_ids = [r["tweet_id"] for r in results_excluded]
    assert "101" not in retrieved_ids, "Leakage guard failed: excluded ID was retrieved!"


def test_escalation_always_escalate_intents():
    """Verify always-escalate intents route to human agents regardless of scores."""
    for intent in config.ALWAYS_ESCALATE_INTENTS:
        res = decide(intent=intent, confidence=0.99, top_similarity=0.99)
        assert res["decision"] == "escalate"
        assert "Always-escalate" in res["reason"]


def test_escalation_low_confidence():
    """Verify queries with confidence below threshold are escalated."""
    low_conf = config.CONFIDENCE_ESCALATION_THRESHOLD - 0.05
    res = decide(intent="order_status", confidence=low_conf, top_similarity=0.80)
    assert res["decision"] == "escalate"
    assert "below threshold" in res["reason"]


def test_escalation_low_similarity():
    """Verify queries with precedent similarity below floor are escalated."""
    low_sim = config.RETRIEVAL_SIMILARITY_FLOOR - 0.05
    res = decide(intent="order_status", confidence=0.90, top_similarity=low_sim)
    assert res["decision"] == "escalate"
    assert "below floor" in res["reason"]


def test_escalation_auto_handle():
    """Verify queries meeting all criteria are routed to automated handling."""
    res = decide(
        intent="order_status",
        confidence=config.CONFIDENCE_ESCALATION_THRESHOLD + 0.10,
        top_similarity=config.RETRIEVAL_SIMILARITY_FLOOR + 0.10
    )
    assert res["decision"] == "auto_handle"
    assert "Sufficient confidence" in res["reason"]
