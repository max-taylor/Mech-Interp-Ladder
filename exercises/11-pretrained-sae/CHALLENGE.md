# 11 — Use a pretrained SAE
**Challenge:** Load a pretrained GPT-2 small residual SAE through SAELens. Pick a feature, collect its top-activating examples, write a one-line interpretation, then steer generation by adding its decoder direction.
**Done when:** you judge that steering clearly shifts the text toward the concept. The check needs `results.json` with ≥ 5 `top_examples`, an `interpretation`, steered text that differs from unsteered, and higher feature activation on the steered text (re-encoded without steering).
**Blind spots to ask an LLM about:** `model_from_pretrained_kwargs`, decoder vs encoder directions, steering coefficient scale, Neuronpedia, feature splitting.
**Stretch:** Sweep the coefficient and plot activation against perplexity.
