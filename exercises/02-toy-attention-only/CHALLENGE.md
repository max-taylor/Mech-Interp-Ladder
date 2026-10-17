# 02 — Train a toy attention-only model
**Challenge:** Train a 2-layer, attention-only transformer on sequences of random tokens repeated twice. Write `results.json` with `vocab_size`, `first_half_loss` and `second_half_loss`.
**Done when:** second-half loss < 0.5 nats while first-half loss stays within 5% of log(vocab). Exclude the first repeated token from the second half.
**Blind spots to ask an LLM about:** `HookedTransformerConfig` with `attn_only`, per-token loss alignment (logits[i] predicts token i+1), why one layer can't do induction, previous-token heads, K-composition.
**Stretch:** Plot both layers' attention patterns and find the previous-token head and the induction head.
