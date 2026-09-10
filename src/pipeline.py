"""
src/pipeline.py - End-to-end execution pipeline for the Customer Support Agent.

Coordinates data loading, pair generation, retrieval indexing, intent classification,
precedent-grounded reply drafting, and escalation gate decision-making.
"""

import argparse
import logging
from pathlib import Path
from typing import Optional
import pandas as pd

from src import config
from src.classifier import IntentClassifier
from src.data_prep import build_brand_pairs, load_raw
from src.escalation import decide
from src.llm_client import LLMClient
from src.reply_gen import ReplyGenerator
from src.retrieval import ResolutionRetriever

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def run(
    input_path: str = "data/amazonhelp_raw.csv",
    brand: str = config.BRAND,
    out_path: str = "outputs/pipeline_output.csv",
    max_rows: int = 5000,
    api_key: Optional[str] = None
) -> pd.DataFrame:
    """
    Runs the complete agent pipeline on customer queries.
    """
    logger.info(f"--- Starting Customer Support Agent Pipeline [{brand}] ---")
    logger.info(f"Loading raw dataset from {input_path}...")
    df_raw = load_raw(input_path)
    logger.info(f"Loaded {len(df_raw)} raw rows.")

    logger.info("Building brand resolution pairs for knowledge retrieval...")
    pairs_df = build_brand_pairs(df_raw, brand=brand)
    logger.info(f"Built {len(pairs_df)} resolution pairs.")

    logger.info("Indexing pairs into TF-IDF ResolutionRetriever...")
    retriever = ResolutionRetriever()
    retriever.fit(pairs_df)

    # Extract inbound customer queries to process
    inbound_mask = df_raw["inbound"] if "inbound" in df_raw.columns else (df_raw["author_id"].str.lower() != brand.lower())
    customer_df = df_raw[inbound_mask].copy()

    if customer_df.empty:
        # If no explicit inbound rows, use pairs as input
        customer_df = pairs_df.copy()

    # Cap rows for fast, reproducible execution
    if max_rows and len(customer_df) > max_rows:
        logger.info(f"Subsampling input queries to {max_rows} rows for swift execution...")
        customer_df = customer_df.head(max_rows)

    logger.info(f"Initializing Gemini LLM client (Model: {config.CLASSIFIER_MODEL})...")
    llm_client = LLMClient(model=config.CLASSIFIER_MODEL, api_key=api_key)
    classifier = IntentClassifier(llm_client)
    generator = ReplyGenerator(llm_client)

    logger.info(f"Processing {len(customer_df)} customer queries through agent pipeline...")
    records = []

    for idx, row in customer_df.iterrows():
        tweet_id = str(row.get("tweet_id", idx))
        text = str(row.get("text", "")).strip()
        if not text:
            continue

        # 1. Intent Classification
        clf_result = classifier.classify(text)
        intent = clf_result["intent"]
        confidence = clf_result["confidence"]

        # 2. Retrieval of past precedents
        precedents = retriever.top_k(text, k=config.TOP_K_RETRIEVAL)
        top_sim = precedents[0]["similarity"] if precedents else 0.0
        top_prec_resolution = precedents[0]["resolution_text"] if precedents else ""

        # 3. Reply Generation
        reply = generator.generate(text, intent=intent, precedents=precedents)

        # 4. Escalation Gate Decision
        gate_result = decide(intent=intent, confidence=confidence, top_similarity=top_sim)

        records.append({
            "tweet_id": tweet_id,
            "text": text,
            "predicted_intent": intent,
            "confidence": confidence,
            "top_similarity": top_sim,
            "top_precedent_resolution": top_prec_resolution,
            "draft_reply": reply,
            "decision": gate_result["decision"],
            "decision_reason": gate_result["reason"]
        })

        if len(records) % 500 == 0 and len(records) > 0:
            logger.info(f"Processed {len(records)}/{len(customer_df)} items...")

    results_df = pd.DataFrame(records)

    # Save outputs
    output_file = Path(out_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    results_df.to_csv(output_file, index=False)
    logger.info(f"Successfully saved {len(results_df)} pipeline results to {output_file}")

    # Summary diagnostics
    if not results_df.empty:
        decision_counts = results_df["decision"].value_counts().to_dict()
        logger.info(f"Pipeline Decision Summary: {decision_counts}")

    return results_df


def main():
    parser = argparse.ArgumentParser(description="Run Hiver Customer Support Agent Pipeline")
    parser.add_argument("--input", "-i", type=str, default="data/amazonhelp_raw.csv", help="Path to input raw CSV")
    parser.add_argument("--brand", "-b", type=str, default=config.BRAND, help="Target brand name")
    parser.add_argument("--output", "-o", type=str, default="outputs/pipeline_output.csv", help="Path for results CSV")
    parser.add_argument("--max-rows", "-m", type=int, default=5000, help="Max rows to process (default: 5000)")
    args = parser.parse_args()

    run(
        input_path=args.input,
        brand=args.brand,
        out_path=args.output,
        max_rows=args.max_rows
    )


if __name__ == "__main__":
    main()
