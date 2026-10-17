# 07 — Activation patching
**Challenge:** On IOI pairs from `common.data.get_ioi_pairs`, patch the clean residual stream (`resid_pre`) into the corrupted run, one (layer, position) at a time. Measure recovery of the logit difference: 0 = corrupted, 1 = clean.
**Done when:** `results.json` has `tokens`, `name_positions` (where clean and corrupt differ) and a 12×positions `recovery_grid`. Final layer, final position must recover > 95%. Name positions must recover > 80% in layers 0–5.
**Blind spots to ask an LLM about:** noising vs denoising, corruption choices (ABC vs name swap), logit diff vs probability metrics, information moving between positions.
**Stretch:** Repeat with attention and MLP outputs per layer instead of `resid_pre`.
