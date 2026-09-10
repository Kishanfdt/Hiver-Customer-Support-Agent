# Golden Set Sampling & Annotation Methodology

This document details the sampling methodology, distribution, and annotation guidelines used to construct the benchmark dataset ([`eval/golden_set.csv`](file:///c:/Users/ckish/OneDrive/Desktop/Customer%20Support%20Agent/eval/golden_set.csv)).

---

## 1. Sampling Strategy

- **Source Corpus:** Inbound customer tweets directed to `@AmazonHelp` extracted from the Kaggle ThoughtVector dataset (`data/amazonhelp_raw.csv`, containing 67,807 inbound tweets).
- **Filtering & Deduplication:**
  - Length filter: Messages strictly $\ge 30$ characters to exclude uninformative noise or single-word pings.
  - Url pruning: Excluded tweets starting with bare URLs or image links without textual substance.
  - Deduplicated: Retweets and repeated bot blasts were pruned to guarantee unique customer interactions.
- **Stratification:** Stratified across the 7 empirical intent categories with 25 real, distinct examples per category, yielding **175 total evaluated interactions**.

### Class Distribution Table

| Intent Category | Primary Customer Need | Evaluated Samples | Default Action |
| :--- | :--- | :--- | :--- |
| `order_status` | Tracking delays, shipment progress, ETA inquiries | 25 | `auto_handle` |
| `refund_request` | Return confirmations, reimbursement timelines | 25 | `auto_handle` |
| `delivery_issue` | Carrier delivery delays, misdelivered packages | 25 | `auto_handle` |
| `account_access` | Password reset, 2FA/OTP failures, locked accounts | 25 | `escalate` |
| `billing_dispute` | Duplicate charges, unauthorized Prime fees, card deductions | 25 | `escalate` |
| `product_defect` | Damaged goods, broken items, incorrect shipments | 25 | `auto_handle` |
| `general_inquiry` | Business hours, locker pickup, gift wrapping, trade-ins | 25 | `auto_handle` |
| **Total** | **Full Benchmark Dataset** | **175** | **50 Escalate / 125 Auto** |

---

## 2. Labelling Guidelines & Disambiguation Protocol

Every entry was assigned a ground-truth label according to these operational principles:

1. **Strict Escalation Policy (`escalate`):**
   - Any query involving unauthorized financial transactions, credit card disputes, or bank account deductions is assigned `billing_dispute` and labeled `escalate`.
   - Any query involving account credentials, compromised logins, or two-factor authentication tokens is assigned `account_access` and labeled `escalate`.
   - **Reasoning:** Public social media channels cannot legally or securely verify customer PII or process monetary adjustments.

2. **Automated Deflection Policy (`auto_handle`):**
   - Standard logistical queries (`order_status`, `delivery_issue`, `refund_request`, `product_defect`, `general_inquiry`) are labeled `auto_handle` because standard brand operating procedure deflects these inquiries to secure DMs with a 17-digit order number or directs them to self-service portal links (`amzn.to/returns`, `amzn.to/YourOrders`).

3. **Compound Query Tiebreak Hierarchy:**
   - When a customer mentions multiple issues (e.g., "My order is broken and I want an immediate refund of my money"):
     - *Rule:* The actionable intent that requires immediate operational triage takes priority. If a physical return must be initiated, it routes to `product_defect` or `refund_request`.
     - *Security Priority:* If unauthorized charges or account locks are mentioned alongside a delivery delay, the security issue always supersedes and forces `escalate`.

---

## 3. Leakage Guard Confirmation

All 175 source `tweet_id`s in `eval/golden_set.csv` are explicitly excluded from the retrieval candidate pool during evaluation via `ResolutionRetriever(exclude_ids=golden_ids)`. This guarantees that evaluation scores reflect genuine precedent generalization rather than verbatim target lookup.
