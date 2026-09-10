"""
eval/run_eval.py - Evaluation runner comparing Trivial Baseline, Simple Baseline, and Live Agent.

Evaluates intent classification accuracy/macro-F1, escalation precision/recall/F1,
and LLM judge scores (relevance, groundedness, tone).
Crucially enforces retrieval leakage protection by excluding golden set tweet IDs
from the precedent corpus.
"""

import argparse
import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Set
import pandas as pd

from eval.baselines import simple_baseline, trivial_baseline
from eval.llm_judge import ReplyJudge
from eval.metrics import escalation_metrics, intent_metrics
from src import config
from src.classifier import IntentClassifier
from src.data_prep import build_brand_pairs, load_raw
from src.escalation import decide
from src.llm_client import LLMClient
from src.reply_gen import ReplyGenerator
from src.retrieval import ResolutionRetriever

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def run_evaluation(
    golden_path: str = "eval/golden_set.csv",
    corpus_path: str = "data/amazonhelp_raw.csv",
    output_path: str = "outputs/eval_report.json",
    brand: str = config.BRAND,
    max_judge_samples: int = 20,
    max_samples: Optional[int] = None,
    offline: bool = False
) -> Dict[str, Any]:
    """
    Executes comparative evaluation across baselines and live agent.
    """
    logger.info(f"Loading golden set benchmark from {golden_path}...")
    golden_df = pd.read_csv(golden_path, dtype={"tweet_id": str})

    if "tweet_id" not in golden_df.columns:
        golden_df["tweet_id"] = [f"gold_{i}" for i in range(len(golden_df))]

    if max_samples and len(golden_df) > max_samples:
        logger.info(f"Subsampling benchmark to {max_samples} items for rapid evaluation...")
        golden_df = golden_df.head(max_samples)

    golden_ids: Set[str] = set(golden_df["tweet_id"].astype(str).tolist())
    logger.info(f"Loaded {len(golden_df)} golden evaluation items. Golden IDs count: {len(golden_ids)}")

    y_gold_intent: List[str] = golden_df["gold_intent"].astype(str).tolist()
    y_gold_decision: List[str] = golden_df["gold_decision"].astype(str).tolist()

    # Load corpus and fit retriever with leakage protection
    logger.info(f"Loading precedent resolution corpus from {corpus_path}...")
    corpus_raw = load_raw(corpus_path)
    pairs_df = build_brand_pairs(corpus_raw, brand=brand)
    logger.info(f"Built {len(pairs_df)} resolution pairs for retrieval corpus.")

    retriever = ResolutionRetriever()
    retriever.fit(pairs_df)

    # Initialize live agent components
    llm_client = LLMClient(model=config.CLASSIFIER_MODEL)
    if offline:
        logger.info("Running evaluation in deterministic offline mode.")
        llm_client.is_offline = True
        llm_client.client = None

    classifier = IntentClassifier(llm_client)
    generator = ReplyGenerator(llm_client)
    judge = ReplyJudge(llm_client)

    # Containers for predictions
    trivial_intents, trivial_decisions = [], []
    simple_intents, simple_decisions = [], []
    agent_intents, agent_decisions, agent_replies = [], [], []
    judge_results = []

    logger.info("Executing evaluation runs across models...")
    for idx, row in golden_df.iterrows():
        text = str(row["text"]).strip()
        t_id = str(row["tweet_id"])

        # 1. Trivial Baseline
        t_res = trivial_baseline(text)
        trivial_intents.append(t_res["predicted_intent"])
        trivial_decisions.append(t_res["decision"])

        # 2. Simple Baseline
        s_res = simple_baseline(text)
        simple_intents.append(s_res["predicted_intent"])
        simple_decisions.append(s_res["decision"])

        # 3. Live Agent (Guarded against retrieval leakage)
        c_res = classifier.classify(text)
        intent = c_res["intent"]
        conf = c_res["confidence"]

        # EXCLUDE golden tweet IDs to guarantee no retrieval leakage
        precedents = retriever.top_k(text, k=config.TOP_K_RETRIEVAL, exclude_ids=golden_ids)
        top_sim = precedents[0]["similarity"] if precedents else 0.0

        # Live generation & judging on representative sample to stay within free-tier limits (< 2 min total runtime)
        if idx < max_judge_samples:
            reply = generator.generate(text, intent=intent, precedents=precedents)
            j_eval = judge.evaluate(customer_tweet=text, generated_reply=reply, precedents=precedents)
            judge_results.append(j_eval)
        else:
            reply = generator._fallback_reply(intent, precedents)

        gate_res = decide(intent=intent, confidence=conf, top_similarity=top_sim)

        agent_intents.append(intent)
        agent_decisions.append(gate_res["decision"])
        agent_replies.append(reply)

        if (idx + 1) % 25 == 0 or (idx + 1) == len(golden_df):
            logger.info(f"Evaluated {idx + 1}/{len(golden_df)} benchmark samples...")

    # Compute metrics
    trivial_intent_m = intent_metrics(y_gold_intent, trivial_intents)
    trivial_esc_m = escalation_metrics(y_gold_decision, trivial_decisions)

    simple_intent_m = intent_metrics(y_gold_intent, simple_intents)
    simple_esc_m = escalation_metrics(y_gold_decision, simple_decisions)

    agent_intent_m = intent_metrics(y_gold_intent, agent_intents)
    agent_esc_m = escalation_metrics(y_gold_decision, agent_decisions)

    # Judge aggregates
    avg_relevance = round(sum(r["relevance"] for r in judge_results) / len(judge_results), 2) if judge_results else 0.0
    avg_groundedness = round(sum(r["groundedness"] for r in judge_results) / len(judge_results), 2) if judge_results else 0.0
    avg_tone = round(sum(r["tone"] for r in judge_results) / len(judge_results), 2) if judge_results else 0.0

    report = {
        "benchmark_metadata": {
            "num_golden_samples": len(golden_df),
            "retrieval_corpus_pairs": len(pairs_df),
            "leakage_guard_excluded_ids": len(golden_ids),
            "brand": brand,
            "offline_mode": llm_client.is_offline
        },
        "models": {
            "trivial_baseline": {
                "intent_metrics": trivial_intent_m,
                "escalation_metrics": trivial_esc_m
            },
            "simple_baseline": {
                "intent_metrics": simple_intent_m,
                "escalation_metrics": simple_esc_m
            },
            "live_agent": {
                "intent_metrics": agent_intent_m,
                "escalation_metrics": agent_esc_m,
                "judge_metrics": {
                    "avg_relevance": avg_relevance,
                    "avg_groundedness": avg_groundedness,
                    "avg_tone": avg_tone
                }
            }
        }
    }

    # Print summary table to stdout
    print("\n" + "=" * 80)
    print("               EVALUATION BENCHMARK RESULTS SUMMARY")
    print("=" * 80)
    print(f"{'Model':<20} | {'Intent Acc':<11} | {'Intent F1':<10} | {'Esc Prec':<9} | {'Esc Rec':<8} | {'Esc F1':<7}")
    print("-" * 80)
    for model_name, m_data in [
        ("Trivial Baseline", report["models"]["trivial_baseline"]),
        ("Simple Baseline", report["models"]["simple_baseline"]),
        ("Live Agent", report["models"]["live_agent"])
    ]:
        i_m = m_data["intent_metrics"]
        e_m = m_data["escalation_metrics"]
        print(
            f"{model_name:<20} | "
            f"{i_m['accuracy']:<11.4f} | "
            f"{i_m['macro_f1']:<10.4f} | "
            f"{e_m['precision']:<9.4f} | "
            f"{e_m['recall']:<8.4f} | "
            f"{e_m['f1']:<7.4f}"
        )
    print("-" * 80)
    print(f"Live Agent LLM Judge Ratings: Relevance={avg_relevance}/5.0, Groundedness={avg_groundedness}/5.0, Tone={avg_tone}/5.0")
    print("=" * 80 + "\n")

    # Save to disk
    out_file = Path(output_path)
    out_file.parent.mkdir(parents=True, exist_ok=True)
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    logger.info(f"Evaluation report successfully saved to {out_file}")

    return report


def main():
    parser = argparse.ArgumentParser(description="Run Evaluation Suite for Customer Support Agent")
    parser.add_argument("--golden", "-g", type=str, default="eval/golden_set.csv", help="Path to golden set CSV")
    parser.add_argument("--corpus", "-c", type=str, default="data/amazonhelp_raw.csv", help="Path to raw corpus CSV")
    parser.add_argument("--output", "-o", type=str, default="outputs/eval_report.json", help="Path for JSON output")
    parser.add_argument("--brand", "-b", type=str, default=config.BRAND, help="Brand identifier")
    parser.add_argument("--max-judge-samples", "-j", type=int, default=20, help="Number of samples to evaluate with LLM judge (default: 20)")
    parser.add_argument("--max-samples", "-m", type=int, default=None, help="Cap total golden set evaluation items (default: all)")
    parser.add_argument("--offline", action="store_true", help="Run evaluation strictly with deterministic offline stubs (zero API calls)")
    args = parser.parse_args()

    run_evaluation(
        golden_path=args.golden,
        corpus_path=args.corpus,
        output_path=args.output,
        brand=args.brand,
        max_judge_samples=args.max_judge_samples,
        max_samples=args.max_samples,
        offline=args.offline
    )


if __name__ == "__main__":
    main()
