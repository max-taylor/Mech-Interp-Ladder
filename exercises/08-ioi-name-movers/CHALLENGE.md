# 08 — IOI circuit slice: name movers
**Challenge:** Use head-level patching on IOI to find the heads that copy the indirect object's name into the output. Ablate them by swapping in their corrupted-run output at the final position (resample ablation).
**Done when:** `results.json` has a 12×12 `head_patching_scores` grid, `identified_heads`, and clean vs ablated logit difference. Ablation must cut the logit difference by > 50%, and your heads must be known name movers, not other IOI heads.
**Blind spots to ask an LLM about:** name movers vs S-inhibition heads, backup name movers, why zero/mean ablation barely moves the result here, attention to the IO position.
**Stretch:** Zero-ablate your heads instead and explain why the logit difference barely drops.
