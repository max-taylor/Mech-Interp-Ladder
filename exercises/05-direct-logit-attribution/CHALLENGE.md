# 05 — Direct logit attribution
**Challenge:** Split the final logit difference (`" Mary"` minus `" John"`) on an IOI prompt into the direct contributions of every head, every MLP, the embeddings and the biases.
**Done when:** `logit_diff_attribution` returns all components and they sum to the true logit difference within 1e-3. Your top head must be a known one.
**Blind spots to ask an LLM about:** freezing the LayerNorm scale, centring components, folded LayerNorm, `b_O` and `b_U`, direct vs indirect effects.
**Stretch:** Show which heads DLA calls important but ablation doesn't, and explain why.
