# Hiver Customer Support Agent (`hiver-customer-support-agent`)

An intelligent, production-grade customer support agent built for `@AmazonHelp` on Twitter, powered by Google's Gemini API via the modern `google-genai` SDK, sparse TF-IDF precedent retrieval, and an auditable deterministic escalation gate.

This repository includes full end-to-end pipelines, data exploration scripts, automated unit tests, an evaluation harness comparing the agent against two baselines with retrieval data-leakage guards, and a comprehensive technical report scaffold.

---

## 🏛️ System Architecture

```
                                 [ Inbound Customer Tweet ]
                                             │
                                             ▼
                                   ┌───────────────────┐
                                   │  IntentClassifier │
                                   └─────────┬─────────┘
                                             │
                       ┌─────────────────────┴─────────────────────┐
                       ▼                                           ▼
             [ Gemini 1.5 Flash ]                         [ Keyword Fallback ]
           (Few-Shot Classification)                    (Offline Stub Mode)
                       │                                           │
                       └─────────────────────┬─────────────────────┘
                                             │ (Intent & Confidence)
                                             ▼
                                  ┌──────────────────────┐
                                  │ ResolutionRetriever  │ ◄─── Exclude Golden IDs
                                  │ (TF-IDF Precedents)  │      (Leakage Guard)
                                  └──────────┬───────────┘
                                             │ (Top-K Matches & Similarity)
                                             ▼
                                  ┌──────────────────────┐
                                  │    ReplyGenerator    │
                                  │ (< 280 chars, Grounded)
                                  └──────────┬───────────┘
                                             │ (Draft Resolution)
                                             ▼
                                  ┌──────────────────────┐
                                  │    EscalationGate    │
                                  └──────────┬───────────┘
                                             │
              ┌──────────────────────────────┴──────────────────────────────┐
              ▼                                                             ▼
     [ Always-Escalate Intent ]                                    [ Qualified Match ]
   - Billing Dispute                                            - Confidence >= 0.55
   - Account Access                                             - Similarity >= 0.15
   - Low Confidence (< 0.55)                                                │
   - Low Similarity (< 0.15)                                                ▼
              │                                                     ✅ AUTO_HANDLE
              ▼                                                  (Deliver Tweet Reply)
       🚨 ESCALATE
  (Route to Human Agent)
```

---

## 📂 Repository Layout

| Directory / File | Description |
| :--- | :--- |
| `src/config.py` | Single tunable configuration source (intents, thresholds, models, brand). |
| `src/data_prep.py` | Loads raw Twitter customer support data and constructs resolution pairs. |
| `src/llm_client.py` | Google GenAI client (`google-genai`) with JSON parser and deterministic offline stubs. |
| `src/classifier.py` | 7-class intent classifier with few-shot prompting and keyword fallback. |
| `src/retrieval.py` | TF-IDF precedent retriever equipped with `exclude_ids` data-leakage guard. |
| `src/reply_gen.py` | Anti-hallucination reply generator adhering strictly to Twitter's 280-character limit. |
| `src/escalation.py` | Pure-Python deterministic decision logic with structured justification reasons. |
| `src/pipeline.py` | End-to-end batch processing pipeline with CLI arguments. |
| `eval/golden_set.csv` | Hand-labeled benchmark dataset structure for evaluation. |
| `eval/golden_set_notes.md` | Sampling and annotation methodology documentation. |
| `eval/baselines.py` | Implementation of Trivial (majority class) and Simple (keyword) baselines. |
| `eval/metrics.py` | Computation of Intent Accuracy/F1 and Escalation Precision/Recall/F1. |
| `eval/llm_judge.py` | LLM-as-a-Judge grading relevance, groundedness, and tone + Cohen's Kappa. |
| `eval/run_eval.py` | Comparative evaluation runner producing `outputs/eval_report.json`. |
| `scripts/download_data.py`| Standalone script downloading Kaggle dataset and filtering `@AmazonHelp`. |
| `scripts/explore_intents.py`| TF-IDF + KMeans clustering script empirically defending the 7 intents. |
| `data/amazonhelp_raw.csv` | Pre-filtered AmazonHelp support dataset (committed for instant reproduction). |
| `data/intent_exploration_notes.md` | Empirical evidence trail and cluster breakdown for the taxonomy. |
| `report/REPORT.md` | 6-page comprehensive technical report scaffold. |
| `DECISION_LOG.md` | 15 architectural and methodological decisions and trade-offs. |
| `CITATIONS.md` | Disclosure of AI assistance (Antigravity) and third-party references. |
| `tests/test_pipeline.py` | Automated offline unit test suite. |

---

## ⚡ Quickstart: Reproduce in Under 15 Minutes

The repository is pre-configured with a committed subset of `data/amazonhelp_raw.csv` and deterministic offline fallback stubs. **Graders can run and test everything immediately without needing a Kaggle token or paid API key.**

### 1. Environment Setup

```bash
# Clone the repository
git clone https://github.com/Kishanfdt/Hiver-Customer-Support-Agent.git
cd Hiver-Customer-Support-Agent

# Create and activate a virtual environment
python -m venv .venv
# On Windows:
.\.venv\Scripts\activate
# On Linux/macOS:
# source .venv/bin/activate

# Install dependencies (uses the current google-genai SDK, not deprecated google-generativeai)
pip install -r requirements.txt
```

> [!NOTE]
> We use the modern `google-genai` SDK (`pip install google-genai`). Do **not** use the deprecated `google-generativeai` package.

### 2. Configure Google Gemini API Key (Optional for Live Mode)

To run in live LLM mode, obtain a free API key from [Google AI Studio](https://aistudio.google.com/apikey):

```bash
# Windows PowerShell:
$env:GOOGLE_API_KEY="your-gemini-api-key"

# Linux / macOS:
export GOOGLE_API_KEY="your-gemini-api-key"
```

*If no key is provided, the system automatically falls back to deterministic offline stubs with zero errors and zero cost.*

### 3. Run Offline Unit Tests

```bash
pytest tests/test_pipeline.py -v
```

### 4. Execute the End-to-End Pipeline

Processes customer queries, retrieves past precedents, drafts grounded replies, and applies the escalation gate:

```bash
python -m src.pipeline --input data/amazonhelp_raw.csv --output outputs/pipeline_output.csv --max-rows 500
```

### 5. Run the Comparative Benchmark Evaluation

Runs the Trivial Baseline, Simple Baseline, and Live Agent (with retrieval data-leakage guards) against the golden benchmark:

```bash
python -m eval.run_eval --golden eval/golden_set.csv --corpus data/amazonhelp_raw.csv --output outputs/eval_report.json
```

---

## 📊 Comparative Baselines

The system is benchmarked against two non-LLM baselines:
1. **Trivial Baseline (`trivial_baseline`):**
   - Predicts the empirical majority-class intent (`order_status`) for every inquiry.
   - Never escalates (`auto_handle`), serving as a lower bound for escalation recall.
2. **Simple Baseline (`simple_baseline`):**
   - Uses domain keyword rules for intent classification.
   - Applies a static mapping where sensitive categories (`billing_dispute`, `account_access`) escalate and all others are auto-handled.

---

## 📥 Full Dataset Ingestion & Intent Exploration (One-Time Setup)

### Download Full Kaggle Dataset
To download the complete multi-million row Kaggle dataset and rebuild `data/amazonhelp_raw.csv`:
1. Ensure your Kaggle API key is saved at `~/.kaggle/kaggle.json` (or set `KAGGLE_USERNAME` and `KAGGLE_KEY`).
2. Run:
   ```bash
   python scripts/download_data.py
   ```

### Empirically Verify Intent Clusters
To verify the evidence trail for our 7 configured intent categories:
```bash
python scripts/explore_intents.py
```
This performs TF-IDF vectorization and $k$-Means clustering on inbound customer queries and updates `data/intent_exploration_notes.md`.

---

## 🛡️ Retrieval Leakage Guard

To prevent synthetic score inflation during evaluation:
- In `eval/run_eval.py`, the `ResolutionRetriever` is supplied with `exclude_ids` containing all tweet IDs present in `eval/golden_set.csv`.
- This ensures the model cannot "cheat" by retrieving the exact held-out resolution from the knowledge base, guaranteeing realistic evaluation metrics.

---

## 🤖 AI Assistance Disclosure

This repository was constructed with AI coding assistance (Antigravity). Full details and references are logged in [CITATIONS.md](CITATIONS.md).
