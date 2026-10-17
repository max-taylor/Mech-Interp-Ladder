# 10 — Train a small SAE (GPU recommended)
**Challenge:** Train a sparse autoencoder on a GPT-2 small mid-layer residual stream: at least 1M tokens, dictionary 8× wider than `d_model`.
**Done when:** `results.json` reports `d_model`, `d_sae`, `n_train_tokens`, mean `l0` < 50, and `spliced_loss` within 15% of `clean_loss` on held-out text, with the SAE reconstruction spliced in.
**Blind spots to ask an LLM about:** L1 vs TopK sparsity, dead latents, decoder norm constraints, activation scaling, the BOS token's huge norm, fraction of variance explained.
**Stretch:** Count dead latents, then add an auxiliary loss (AuxK) or resampling to revive them.
