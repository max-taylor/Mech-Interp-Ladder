# 04 — Logit lens
**Challenge:** For each prompt in `common.data.FACTUAL_PROMPTS`, decode the final-position residual stream after the embeddings and after every block through the final LayerNorm and unembedding.
**Done when:** `results.json` lists, per prompt, `top_token_by_layer` (13 strings) and `first_correct_layer` (or null). The check confirms your last depth equals the model's own prediction.
**Blind spots to ask an LLM about:** residual stream view, `ln_final` at intermediate layers, tuned lens, why early layers predict bigram-ish tokens.
**Stretch:** Plot the answer's rank or probability by layer for every prompt on one chart.
