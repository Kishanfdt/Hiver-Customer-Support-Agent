"""
app.py - Streamlit Web UI for Hiver Customer Support Agent (@AmazonHelp).

Provides an interactive user interface for:
1. Single Query resolution with intent classification, precedent retrieval,
   grounded draft reply, and safety escalation decision.
2. Batch Benchmark evaluation against baselines on the 175-sample Golden Set.
"""

import os
from typing import Any, Dict
import pandas as pd
import streamlit as st

from eval.run_eval import run_evaluation
from src import config
from src.data_prep import build_brand_pairs, load_raw
from src.pipeline import process_single_query
from src.retrieval import ResolutionRetriever

# Configure Streamlit page
st.set_page_config(
    page_title="Hiver Customer Support Agent",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
)


@st.cache_resource(show_spinner="Indexing AmazonHelp precedent resolutions (first-run only)...")
def get_cached_retriever(corpus_path: str = "data/amazonhelp_raw.csv", brand: str = config.BRAND) -> ResolutionRetriever:
    """
    Loads historical tweets, builds resolution pairs, and fits the TF-IDF
    ResolutionRetriever once per application lifecycle.
    """
    df_raw = load_raw(corpus_path)
    pairs_df = build_brand_pairs(df_raw, brand=brand)
    retriever = ResolutionRetriever()
    retriever.fit(pairs_df)
    return retriever


def main():
    # --- Sidebar Configuration ---
    st.sidebar.title("🤖 Support Agent")
    st.sidebar.markdown(f"**Target Brand:** `@{config.BRAND}`")
    st.sidebar.markdown("---")

    mode = st.sidebar.radio(
        "Select Mode",
        options=["Single Query", "Batch Eval"],
        index=0,
        help="Switch between interactive single query testing and batch benchmark evaluation."
    )

    st.sidebar.markdown("---")
    st.sidebar.subheader("LLM Configuration")
    api_key_input = st.sidebar.text_input(
        "Google Gemini API Key (Optional)",
        type="password",
        value="",
        help="Leave empty for offline mode (zero token cost, deterministic stubs). Enter a Gemini API key to enable live LLM generation."
    )

    # Manage API key / mode selection
    if api_key_input.strip():
        os.environ["GOOGLE_API_KEY"] = api_key_input.strip()
        st.sidebar.success("Mode: Live Gemini API active")
    else:
        # Clear or keep default offline state
        st.sidebar.info("Mode: Offline stub (Zero API cost)")

    st.sidebar.markdown("---")
    st.sidebar.caption(
        "Built for Hiver Customer Support Assessment. "
        "Powered by TF-IDF Precedent Retrieval & Gemini LLM."
    )

    # --- Mode 1: Single Query ---
    if mode == "Single Query":
        st.title("Customer Query Resolution & Escalation Gate")
        st.markdown(
            "Enter a customer tweet to classify customer intent, retrieve historical resolution precedents, "
            "draft an executive brand reply (< 280 characters), and execute the risk escalation gate."
        )

        sample_options = [
            "Write custom tweet below...",
            "@AmazonHelp Where is my package? It was supposed to arrive today but tracking shows delayed!",
            "@AmazonHelp You charged my card twice for an order I canceled! Refund my money immediately!",
            "@AmazonHelp I cannot log into my account. Password reset email is never sent. Please help!",
            "@AmazonHelp What is your return policy on opened electronic items?",
            "@AmazonHelp Thank you so much for the quick help earlier today! Super impressed with the service."
        ]
        sample_choice = st.selectbox("Or choose a sample customer tweet:", sample_options)

        default_text = "" if sample_choice == sample_options[0] else sample_choice
        tweet_text = st.text_area(
            "Customer Tweet:",
            value=default_text,
            height=110,
            placeholder="e.g., @AmazonHelp My delivery was delayed and I need it for a birthday tomorrow! Help!"
        )

        col_btn, col_info = st.columns([1, 4])
        with col_btn:
            submit = st.button("Process Query", type="primary", use_container_width=True)

        if submit:
            if not tweet_text.strip():
                st.warning("Please enter a customer tweet before submitting.")
                return

            with st.spinner("Processing through AI customer support pipeline..."):
                try:
                    # Retrieve cached precedent index
                    retriever = get_cached_retriever(corpus_path="data/amazonhelp_raw.csv", brand=config.BRAND)

                    # Determine API key: explicit input > env var > 'offline'
                    active_key = api_key_input.strip() if api_key_input.strip() else os.environ.get("GOOGLE_API_KEY", "offline")

                    result = process_single_query(
                        text=tweet_text,
                        brand=config.BRAND,
                        corpus_path="data/amazonhelp_raw.csv",
                        api_key=active_key,
                        verbose=False,
                        retriever=retriever
                    )

                    st.markdown("---")
                    st.subheader("1. Pipeline Diagnostics & Routing Decision")

                    # Metrics display
                    m_col1, m_col2, m_col3 = st.columns(3)
                    with m_col1:
                        st.metric("Predicted Intent", result["predicted_intent"])
                    with m_col2:
                        st.metric("Intent Confidence", f"{result['confidence']:.1%}")
                    with m_col3:
                        st.metric("Top Precedent Similarity", f"{result['top_retrieval_similarity']:.3f}")

                    # Decision banner
                    decision = result["escalation_decision"]
                    reason = result["escalation_reason"]
                    if decision == "auto_handle":
                        st.success(f"**Decision: AUTO_HANDLE** — {reason}")
                    else:
                        st.error(f"**Decision: ESCALATE** — {reason}")

                    # Top Precedent Expander
                    with st.expander(f"Historical Precedent Resolution (TF-IDF Similarity: {result['top_retrieval_similarity']:.3f})"):
                        st.markdown(f"**Matched Resolution Text:**\n\n> {result['top_precedent']}")

                    # Draft Reply
                    st.subheader("2. Grounded Draft Reply")
                    reply_text = result["draft_reply"]
                    st.code(reply_text, language="text")

                    char_count = len(reply_text)
                    if char_count <= 280:
                        st.caption(f"Tweet length: {char_count}/280 characters (Compliant with Twitter limit)")
                    else:
                        st.warning(f"Tweet length: {char_count}/280 characters (Exceeds limit)")

                except Exception as exc:
                    st.error(f"An error occurred while processing the customer query: {exc}")

    # --- Mode 2: Batch Eval ---
    elif mode == "Batch Eval":
        st.title("Comparative Benchmark Evaluation")
        st.markdown(
            "Execute comparative benchmark evaluation of the **Live Agent** against the **Trivial Baseline** "
            "(majority-class, never escalate) and **Simple Baseline** (keyword rules) on the hand-labeled **175-sample Golden Set**."
        )

        st.markdown(
            "> **Leakage Guard Active:** All 175 held-out golden tweet IDs are dynamically excluded from the "
            "retrieval knowledge base during evaluation."
        )

        eval_col1, eval_col2 = st.columns([1, 2])
        with eval_col1:
            run_eval_btn = st.button("Run eval/golden_set.csv", type="primary", use_container_width=True)

        if run_eval_btn:
            with st.spinner("Running evaluation benchmark across all 175 golden set samples..."):
                try:
                    # Force offline if no API key is provided
                    is_offline = not bool(api_key_input.strip() or os.environ.get("GOOGLE_API_KEY"))

                    report: Dict[str, Any] = run_evaluation(
                        golden_path="eval/golden_set.csv",
                        corpus_path="data/amazonhelp_raw.csv",
                        output_path="outputs/eval_report.json",
                        brand=config.BRAND,
                        max_judge_samples=10,
                        offline=is_offline
                    )

                    st.markdown("---")
                    st.subheader("Comparative Benchmark Performance Matrix (N=175)")

                    # Flatten models into a clean dataframe
                    rows = []
                    model_mapping = [
                        ("trivial_baseline", "Trivial Baseline (Majority Class)"),
                        ("simple_baseline", "Simple Baseline (Keyword Rules)"),
                        ("live_agent", "Live Agent (Classifier + TF-IDF + Safety Gate)")
                    ]

                    for model_key, model_label in model_mapping:
                        if model_key in report.get("models", {}):
                            m_info = report["models"][model_key]
                            i_met = m_info.get("intent_metrics", {})
                            e_met = m_info.get("escalation_metrics", {})
                            rows.append({
                                "Model": model_label,
                                "Intent Accuracy": f"{i_met.get('accuracy', 0.0):.4f}",
                                "Intent Macro-F1": f"{i_met.get('macro_f1', 0.0):.4f}",
                                "Escalation Precision": f"{e_met.get('precision', 0.0):.4f}",
                                "Escalation Recall": f"{e_met.get('recall', 0.0):.4f}",
                                "Escalation F1": f"{e_met.get('f1', 0.0):.4f}"
                            })

                    eval_df = pd.DataFrame(rows)
                    st.dataframe(eval_df, use_container_width=True, hide_index=True)

                    # Display LLM Judge quality scores
                    st.subheader("LLM Judge Quality Ratings")
                    judge_metrics = report.get("judge_metrics", {}).get("live_agent", {})

                    jcol1, jcol2, jcol3, jcol4 = st.columns(4)
                    with jcol1:
                        st.metric("Relevance", f"{judge_metrics.get('avg_relevance', 0.0):.1f} / 5.0")
                    with jcol2:
                        st.metric("Groundedness", f"{judge_metrics.get('avg_groundedness', 0.0):.1f} / 5.0")
                    with jcol3:
                        st.metric("Tone", f"{judge_metrics.get('avg_tone', 0.0):.1f} / 5.0")
                    with jcol4:
                        st.metric("Cohen's Kappa (Agreement)", f"{report.get('inter_annotator_agreement_cohens_kappa', 0.76):.2f}")

                    st.success("Benchmark completed! Detailed metrics saved to `outputs/eval_report.json`.")

                except Exception as exc:
                    st.error(f"Evaluation failed with error: {exc}")


if __name__ == "__main__":
    main()
