# Technical Evaluation Report: Hiver Customer Support Agent (`AmazonHelp`)

**Author:** [Candidate Name]  
**Date:** September 2026  
**Repository:** [https://github.com/Kishanfdt/Hiver-Customer-Support-Agent](https://github.com/Kishanfdt/Hiver-Customer-Support-Agent)  
**Maximum Length:** 6 Pages

---

## 1. Problem Framing & System Boundaries

### What "Good" Means for AmazonHelp on Twitter
In social media customer service, especially for high-volume consumer retail brands like Amazon:
1. **Response Velocity & Brand Safety:** Automated triage must respond promptly without committing to unauthorized refunds, promises, or policies that create corporate liability.
2. **Strict Escalation Compliance:** Queries involving sensitive personal authentication (`account_access`) or payment disputes (`billing_dispute`) must never be resolved over public Twitter; they must be immediately routed to secure authenticated channels or human specialists.
3. **Factual Groundedness:** Auto-generated responses must strictly reflect official precedent and established procedures rather than inventing hallucinated ETAs or compensation amounts.

### What We Chose NOT to Build (Non-Goals)
- **Autonomous Financial Execution:** The system is explicitly forbidden from executing direct refunds or modifying user accounts via API.
- **Unbounded Conversational Agent:** The agent is not designed as an open-ended conversational companion; every reply is constrained to under 280 characters and targeted toward issue triage or secure DM deflection.
- **Complex Multi-Turn State Machine:** Because Twitter support interactions typically initiate with public triage followed by direct message handoff, the agent focuses on high-precision single-turn deflection and routing.

---

## 2. Experimental Results vs. Two Baselines

### Quantitative Benchmark Comparison
The agent was benchmarked against a hand-labeled golden set (`eval/golden_set.csv`) evaluated against two distinct non-LLM baselines:
1. **Trivial Baseline:** Always predicts majority-class intent (`order_status`) and never escalates (`auto_handle`).
2. **Simple Baseline:** Keyword-rule intent classification with fixed intent-to-decision routing.
3. **Live Agent:** Gemini 1.5 Flash few-shot classification + TF-IDF precedent retrieval (with leakage exclusion) + dual-gate escalation.

| System | Intent Accuracy | Intent Macro-F1 | Escalation Precision | Escalation Recall | Escalation F1 |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Trivial Baseline** | [Fill from eval] | [Fill from eval] | 0.0000 | 0.0000 | 0.0000 |
| **Simple Baseline** | [Fill from eval] | [Fill from eval] | [Fill from eval] | [Fill from eval] | [Fill from eval] |
| **Live Agent (Gemini)** | [Fill from eval] | [Fill from eval] | [Fill from eval] | [Fill from eval] | [Fill from eval] |

### Generation Quality (LLM-as-a-Judge)
Automated evaluation using rubric scoring (1 to 5 scale) on held-out customer interactions:
- **Relevance:** [Fill / 5.0]
- **Groundedness / Factuality:** [Fill / 5.0]
- **Brand Tone & Empathy:** [Fill / 5.0]
- **Judge-Human Agreement (Cohen's Kappa):** [Fill $\kappa$]

---

## 3. Failure Analysis (Top 5 Failure Modes)

*Document the primary error patterns observed during testing with concrete examples:*

1. **Failure Mode 1: Conflation of Delivery Delay and Lost Package**
   - *Customer Tweet:* "[Example]"
   - *Predicted Intent:* `order_status` vs. *Ground Truth:* `delivery_issue`
   - *Impact & Root Cause:* [Why the classifier or retriever stumbled and how to address it]

2. **Failure Mode 2: Multi-Intent Queries with Compound Complaints**
   - *Customer Tweet:* "[Example]"
   - *Issue:* Customer mentions a defective product and demands an instant refund.
   - *Impact & Root Cause:* Single-label classification assigns `refund_request` while precedent retrieval matches `product_defect`.

3. **Failure Mode 3: Low Retrieval Similarity on Idiosyncratic Phrasing**
   - *Customer Tweet:* "[Example]"
   - *Issue:* Unusual vocabulary or slang caused TF-IDF cosine similarity to fall below the 0.15 threshold, causing an unnecessary escalation.

4. **Failure Mode 4: Sarcasm and Negative Sentiment Masking Intent**
   - *Customer Tweet:* "[Example]"
   - *Issue:* Heavy irony obscuring the underlying issue (e.g. "Great job delivering my package to the roof!").

5. **Failure Mode 5: Boundary Ambiguity Between Billing and Cancellation**
   - *Customer Tweet:* "[Example]"
   - *Issue:* Customer asking why a cancelled item is still showing a pending card authorization.

---

## 4. Mandatory Section: What Is Misleading About My Headline Number?

> [!CAUTION]
> Every benchmark metric can conceal underlying systemic distortions. Below is an honest appraisal of potential risks and biases in our headline evaluation metrics:

### Candidate Vulnerability 1: Retrieval Data Leakage Risk
- If an evaluation harness searches a retrieval database containing the exact held-out customer queries or their associated brand responses, cosine similarity will artificially approach 1.0, and the reply generator will merely copy the historical answer.
- **How we guarded against this:** We engineered explicit `exclude_ids` filtering in `ResolutionRetriever.top_k`, guaranteeing that no golden set item or its source tweet ID can appear in the candidate retrieval pool during evaluation.

### Candidate Vulnerability 2: Golden-Set Sampling Bias & Distribution Shift
- The golden set consists of [150–250] hand-curated examples. If sampled disproportionately from common keywords (e.g. "where is my order"), head distributions will be over-represented while tail anomalies (account takeovers, localized shipping anomalies) will be under-represented.
- A headline accuracy of 85%+ on head distributions may conceal a 40% failure rate on rare or ambiguous edge cases.

### Candidate Vulnerability 3: LLM Judge Leniency & Circularity
- Using an LLM to evaluate LLM-generated responses can introduce positive self-preference bias. While we measured Cohen's kappa against human annotations, automated judge rubrics can still overestimate groundedness when subtle factual policy discrepancies occur.

---

## 5. What I'd Do Next With One More Week

1. **Dense Hybrid Retrieval (BM25 + Dense Embeddings):**
   - Augment sparse TF-IDF with dense vector embeddings (e.g., `text-embedding-004`) to handle semantic paraphrasing, typos, and customer slang that fail keyword matching.
2. **Dynamic Multi-Turn Context Tracking:**
   - Extend the architecture to reconstruct full conversation threads via `in_response_to_tweet_id` chains to handle customer clarifications across multiple turns.
3. **Active Learning & Negative Mining for Threshold Calibration:**
   - Empirically calibrate the dual confidence and similarity thresholds ($0.55$ and $0.15$) via precision-recall curves to optimize the trade-off between human deflection savings and customer churn risk.
4. **Automated Red-Teaming & Policy Safety Guardrails:**
   - Implement dedicated guardrails (e.g. NeMo Guardrails or Llama Guard) to block prompt injection attacks and social engineering attempts in inbound tweets.
