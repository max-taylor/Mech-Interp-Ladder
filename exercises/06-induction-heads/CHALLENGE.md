# 06 — Induction heads in the wild
**Challenge:** Score every GPT-2 small head by how much it attends from a token to the token after its earlier copy, using repeated random tokens. Ablate your top 5, and 5 random others.
**Done when:** `results.json` has a 12×12 `induction_scores` grid, `top_heads` (best first, as `"L5H1"`-style keys), and both loss increases on the repeated half. Your top heads must overlap the known induction heads, and ablating them must hurt at least 2× more than random heads.
**Blind spots to ask an LLM about:** attention-pattern diagonals and offsets, BOS tokens, previous-token heads, induction vs duplicate-token heads.
**Stretch:** Find the previous-token heads feeding them via K-composition scores.
