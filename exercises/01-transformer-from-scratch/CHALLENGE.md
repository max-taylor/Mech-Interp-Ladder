# 01 — Transformer from scratch
**Challenge:** Implement GPT-2 small in `solution.py`: embed, positional embed, LayerNorm, causal attention, MLP, block, unembed. Parameter names and shapes must match TransformerLens so its weights load straight in.
**Done when:** your logits match `HookedTransformer` (unprocessed weights) on a test prompt within 1e-4 (compared in float64).
**Blind spots to ask an LLM about:** pre-LN vs post-LN, causal masking with -inf, einsum/einops index notation, GELU tanh approximation, why `from_pretrained_no_processing`.
**Stretch:** Add a KV cache and greedy sampling; check your generations match `model.generate`.
