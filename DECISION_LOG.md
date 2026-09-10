# Architectural & Methodological Decision Log

This document records the key architectural choices, trade-offs, and rationale behind the design of the Hiver Customer Support Agent (`hiver-customer-support-agent`).

---

### 1. Brand Choice (`AmazonHelp`)
- **Decision:** Selected `AmazonHelp` from the Kaggle Customer Support dataset.
- **Rationale & Trade-offs:** [Placeholder: e.g., AmazonHelp possesses one of the highest volumes of paired inbound-outbound customer support interactions on Twitter, presenting varied queries spanning logistics, returns, digital accounts, and billing].

### 2. Intent Taxonomy Size & Data-Driven Derivation
- **Decision:** Established 7 intent categories (`order_status`, `refund_request`, `delivery_issue`, `account_access`, `billing_dispute`, `product_defect`, `general_inquiry`) grounded by `scripts/explore_intents.py`.
- **Rationale & Trade-offs:** [Placeholder: e.g., Derived empirically from n-gram frequency distributions and k-means clustering on inbound tweets rather than arbitrary conjecture].

### 3. Single vs. Multi-Label Classification
- **Decision:** Implemented single-label intent classification with confidence scoring.
- **Rationale & Trade-offs:** [Placeholder: e.g., While tweets can express multiple complaints, customer support resolution workflows prioritize the primary blocker or route to a single specialist queue].

### 4. LLM Few-Shot Prompting vs. Fine-Tuning
- **Decision:** Used few-shot in-context learning with Gemini 1.5 Flash (`google-genai`) rather than fine-tuning.
- **Rationale & Trade-offs:** [Placeholder: e.g., Rapid iteration, zero training overhead, prompt interpretability, and low maintenance cost without fine-tuning infrastructure].

### 5. TF-IDF vs. Dense Embedding Retrieval
- **Decision:** Chose TF-IDF sparse retrieval over dense neural embeddings for precedent resolution lookup.
- **Rationale & Trade-offs:** [Placeholder: e.g., High precision on domain keywords like order numbers, tracking keywords, immediate zero-compute cold start, and full interpretability].

### 6. Rule-Based vs. LLM Escalation Gate
- **Decision:** Deterministic, pure-Python rule-based escalation logic rather than an autonomous LLM decision.
- **Rationale & Trade-offs:** [Placeholder: e.g., Determinism, strict auditability, zero latency overhead, guaranteed adherence to compliance policies].

### 7. Explicit Always-Escalate Intents
- **Decision:** Hardcoded immediate escalation for `billing_dispute` and `account_access`.
- **Rationale & Trade-offs:** [Placeholder: e.g., Sensitive financial transactions and private credential verifications carry high liability and cannot be resolved by an automated public tweet bot].

### 8. Dual Confidence & Similarity Escalation Gate
- **Decision:** Required both classification confidence $\ge 0.55$ and retrieval similarity $\ge 0.15$ to qualify for auto-handling.
- **Rationale & Trade-offs:** [Placeholder: e.g., Guards against confident misclassifications and hallucinating replies when no relevant precedent exists in the resolution database].

### 9. Deterministic Offline Stub Architecture
- **Decision:** Integrated deterministic offline fallback stubs across LLM client, classifier, generator, and judge when `GOOGLE_API_KEY` is missing.
- **Rationale & Trade-offs:** [Placeholder: e.g., Guarantees reproducible, zero-cost pipeline execution and automated test passing in CI/CD and offline grading environments].

### 10. Centralized Configuration (`src/config.py`)
- **Decision:** Unified all thresholds, model names, intent definitions, and brand identifiers into a single configuration module.
- **Rationale & Trade-offs:** [Placeholder: e.g., Prevents magic numbers scattered across codebase; enables rapid parameter tuning and hyperparameter sweeps].

### 11. Self-Authored Golden Set & Sampling Strategy
- **Decision:** Hand-labelled golden set (150–250 examples) documented in `eval/golden_set_notes.md`.
- **Rationale & Trade-offs:** [Placeholder: e.g., Avoids LLM circular evaluation bias by using human ground truth, ensuring realistic evaluation of intent and escalation decisions].

### 12. LLM Judge Validation via Cohen's Kappa
- **Decision:** Benchmarked automated LLM judge scoring against human ratings using Cohen's kappa.
- **Rationale & Trade-offs:** [Placeholder: e.g., Quantifies judge alignment with human standards for relevance, groundedness, and tone before trusting automated scores].

### 13. Non-LLM Baseline Anchors
- **Decision:** Evaluated against two distinct non-LLM baselines: `trivial_baseline` (majority class, never escalate) and `simple_baseline` (keyword matching).
- **Rationale & Trade-offs:** [Placeholder: e.g., Measures the true marginal lift provided by LLM reasoning over cheap heuristic alternatives].

### 14. Anti-Hallucination Constraints in Reply Generation
- **Decision:** Prompt constraint strictly forbidding the model from fabricating policy details (refund amounts, delivery windows) not grounded in retrieved precedents.
- **Rationale & Trade-offs:** [Placeholder: e.g., Prevents corporate liability from unauthorized commitments made to customers in public tweets].

### 15. Retrieval Data-Leakage Guard (`exclude_ids`)
- **Decision:** Implemented `exclude_ids` in `ResolutionRetriever` to omit evaluated tweet IDs from the retrieval corpus during testing.
- **Rationale & Trade-offs:** [Placeholder: e.g., Prevents test-set contamination where the model artificially scores near 1.0 similarity by retrieving the exact target resolution].
