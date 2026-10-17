# 03 — Hooks and ablation
**Challenge:** Use TransformerLens hooks to zero-ablate and mean-ablate each of GPT-2 small's 144 attention heads, one at a time, on a text sample (`common.data.load_pile_texts`).
**Done when:** `results.json` has `clean_loss` plus `zero_ablation` and `mean_ablation` maps of `"L{layer}H{head}"` → loss increase, and mean ablation is no harsher than zero ablation on average.
**Blind spots to ask an LLM about:** `hook_z` vs `hook_result`, `run_with_hooks`, `names_filter`, why zero is off-distribution, resample ablation.
**Stretch:** Plot both 12×12 grids side by side and explain the heads where they disagree most.
