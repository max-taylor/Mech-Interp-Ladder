# 09 — Linear probes
**Challenge:** Label each token from `common.data.make_quote_texts` as inside or outside quotation marks. Train a logistic probe on the residual stream at every depth. Train a control probe on shuffled labels.
**Done when:** `results.json` has `accuracy_by_layer`, `control_accuracy_by_layer` (held-out, split by text) and `majority_baseline`. Best layer > 85%; the control never beats the baseline by more than 5 points.
**Blind spots to ask an LLM about:** train/test leakage across tokens of one text, feature standardisation, selectivity and control tasks, probing vs causal use.
**Stretch:** Steer with the probe direction and see whether GPT-2 starts closing quotes.
