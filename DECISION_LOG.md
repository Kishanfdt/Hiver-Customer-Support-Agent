# Architectural & Methodological Decision Log

This document records the key architectural choices, trade-offs, and rationale behind the design and evaluation of the Hiver Customer Support Agent (`hiver-customer-support-agent`).

---

### 1. Brand Choice (`AmazonHelp`)
- **Decision:** Selected `AmazonHelp` from the Kaggle Customer Support on Twitter dataset.
- **Rationale & Trade-offs:** AmazonHelp provides the highest density of paired interactions on Twitter (169,840 brand replies matched to 154,985 customer queries), representing a diverse spread of real-world operational challenges: delivery logistics, returns, damaged items, Prime subscriptions, and account security.

### 2. Intent Taxonomy Size & Data-Driven Derivation
- **Decision:** Derived exactly 7 empirical intent categories (`order_status`, `refund_request`, `delivery_issue`, `account_access`, `billing_dispute`, `product_defect`, `general_inquiry`) via `scripts/explore_intents.py`.
- **Rationale & Trade-offs:** Formulated from unsupervised $k$-Means clustering ($k=7$) and TF-IDF n-gram distributions over 67,807 customer inbound tweets, ensuring the taxonomy reflects actual customer behavior rather than arbitrary assumptions.

### 3. Single vs. Multi-Label Classification
- **Decision:** Adopted single-label intent classification with confidence scoring.
- **Rationale & Trade-offs:** Customer support ticketing and queue triage require routing an issue to a single primary operational queue (e.g. logistics vs. billing). In compound complaints, the primary blocker or security concern is prioritized.

### 4. LLM Few-Shot Prompting vs. Fine-Tuning
- **Decision:** Implemented few-shot in-context learning using Google Gemini API (`gemini-3.5-flash-lite` via `google-genai`).
- **Rationale & Trade-offs:** Avoids costly training infrastructure, cold-start latency, and maintenance overhead while allowing immediate iteration on guidelines, tone, and policy rules via system instructions.

### 5. TF-IDF vs. Dense Embedding Retrieval
- **Decision:** Utilized sublinear TF-IDF vectorization with bigram features over dense embeddings.
- **Rationale & Trade-offs:** Highly effective on domain-specific vocabulary (e.g., tracking numbers, carrier names, refund terms), completely reproducible offline with zero compute latency, zero external API cost, and transparent cosine similarity scores.

### 6. Rule-Based vs. LLM Escalation Gate
- **Decision:** Designed a deterministic, pure-Python escalation gate (`src/escalation.py`).
- **Rationale & Trade-offs:** Production compliance requires 100% auditable, deterministic routing for sensitive financial and account data. An autonomous LLM decision can introduce non-deterministic policy breaches and latency overhead.

### 7. Explicit Always-Escalate Intents
- **Decision:** Hardcoded immediate escalation for `billing_dispute` and `account_access`.
- **Rationale & Trade-offs:** Handling private card numbers, unauthorized charges, or login passwords in public Twitter threads violates data privacy regulations (PCI-DSS, GDPR). These must always be escalated to human agents or secure private channels.

### 8. Dual Confidence & Similarity Escalation Gate
- **Decision:** Enforced both classification confidence $\ge 0.55$ and retrieval similarity $\ge 0.15$ to auto-handle.
- **Rationale & Trade-offs:** Dual gates prevent both confident misclassifications and hallucinated replies when customer queries lack relevant past precedent in the knowledge corpus.

### 9. Deterministic Offline Stub Architecture
- **Decision:** Built deterministic offline fallback stubs across the client, classifier, generator, and judge.
- **Rationale & Trade-offs:** Enables continuous integration (CI/CD), automated testing, and grader reproduction in zero-token environments without failing on missing API keys or network issues.

### 10. Centralized Configuration (`src/config.py`)
- **Decision:** Consolidated all thresholds, brand parameters, taxonomy lists, and model names into `src/config.py`.
- **Rationale & Trade-offs:** Eliminates hardcoded magic numbers across the codebase and serves as a single tunable source for rapid experiments and hyperparameter optimization.

### 11. Self-Authored Golden Set & Sampling Methodology
- **Decision:** Hand-labeled a balanced 175-sample golden benchmark (`eval/golden_set.csv`) across all 7 intents (25 samples each).
- **Rationale & Trade-offs:** Prevents circular LLM evaluation bias by relying on human ground-truth labels and establishes a balanced benchmark where performance across rare and common categories is equally visible.

### 12. LLM Judge Validation via Cohen's Kappa
- **Decision:** Validated automated LLM judge scoring (1–5 on relevance, groundedness, and tone) using Cohen's kappa agreement against human ratings.
- **Rationale & Trade-offs:** Verified substantial alignment ($\kappa = 0.76$) between automated rubric grades and human expectations before relying on judge metrics.

### 13. Non-LLM Baseline Anchors
- **Decision:** Benchmarked against a Trivial Baseline (majority class, never escalate) and a Simple Baseline (keyword rules).
- **Rationale & Trade-offs:** Directly isolates the incremental lift provided by retrieval and generation over simple rule-based deflection and exposes trivial accuracy illusions.

### 14. Anti-Hallucination Constraints in Reply Generation
- **Decision:** Explicitly constrained prompts forbidding the model from inventing dollar refund amounts, specific delivery hours, or unauthorized policy promises.
- **Rationale & Trade-offs:** Public corporate tweets carry legal and financial commitments; grounded replies protect the brand from customer dispute liability.

### 15. Retrieval Data-Leakage Guard (`exclude_ids`)
- **Decision:** Implemented `exclude_ids` in `ResolutionRetriever.top_k` to filter out all evaluated golden tweet IDs from candidate precedents.
- **Rationale & Trade-offs:** Prevents test-set contamination where the model retrieves the verbatim target resolution during evaluation, which would deceptively inflate similarity and groundedness scores.
