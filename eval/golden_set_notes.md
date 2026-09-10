# Golden Set Sampling & Annotation Methodology

This document details the exact sampling and labelling protocol used to construct the evaluation benchmark (`eval/golden_set.csv`).

---

## 1. Sampling Strategy
*Document how the 150–250 evaluation examples were sampled from the filtered AmazonHelp dataset.*

- **Data Source:** Raw inbound customer tweets directed to `@AmazonHelp` from the Kaggle dataset (`thoughtvector/customer-support-on-twitter`).
- **Sampling Method:** [To be completed by author: e.g., Stratified sampling across intent keywords / temporal slices vs. Uniform random sampling from the filtered brand set].
- **Candidate Pool Size:** [e.g., Sampled from ~5,000 candidate inbound customer tweets].
- **Deduplication:** Repeated retweets and duplicate customer spam messages were pruned prior to selection.

---

## 2. Labelling Protocol & Guidelines
*Document the human guidelines and rules applied during manual annotation.*

- **Annotator:** Authored directly by human evaluator (not auto-labeled by LLM to avoid circular evaluation bias).
- **Target Fields:**
  1. `text`: Customer's original inbound message.
  2. `gold_intent`: Exactly one of the 7 empirical categories (`order_status`, `refund_request`, `delivery_issue`, `account_access`, `billing_dispute`, `product_defect`, `general_inquiry`).
  3. `gold_decision`: Either `auto_handle` (suitable for automated brand response) or `escalate` (requires human tier-2 support or sensitive verification).

---

## 3. Disambiguation & Tiebreak Rules
*Rules applied when a tweet contained overlapping issues or ambiguous phrasing:*

1. **Billing vs. Refund Conflict:**
   - If the customer complains about an unexpected charge or being billed twice without an existing return, label as `billing_dispute` (`escalate`).
   - If the customer initiated a return and is inquiring about pending reimbursement, label as `refund_request` (`auto_handle`).
2. **Delivery Issue vs. Order Status:**
   - If the carrier has missed the estimated delivery window or marked the item as delivered while missing, label as `delivery_issue`.
   - If the item is still in transit within the normal window and the customer simply requests tracking info, label as `order_status`.
3. **Escalation Priority:**
   - If an inquiry mentions compromised credentials or unauthorized financial deductions, always label as `escalate` regardless of secondary questions.
