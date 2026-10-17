# mech-interp-ladder

Twelve hands-on mechanistic interpretability exercises on GPT-2 small and toy models, in order.

**Philosophy:** build first. Each `CHALLENGE.md` is a goal, a test and a list of blind spots, not a tutorial. Start coding, ask an LLM when you hit a blind spot, keep notes as you go, and iterate until the check passes.

## Setup

```sh
uv sync
```

Runs on CPU, Apple MPS or CUDA (picked automatically by `common/device.py`). The first run downloads GPT-2 small, `NeelNanda/pile-10k` and, for 11, a SAELens SAE.

## The ladder

1. **Transformer from scratch:** GPT-2 small in plain PyTorch, matching logits. ~4h
2. **Toy attention-only model:** train one that learns induction. ~2h
3. **Hooks and ablation:** zero/mean-ablate all 144 heads. ~2h
4. **Logit lens:** decode the residual stream layer by layer. ~1h
5. **Direct logit attribution:** split a logit diff across components. ~2h
6. **Induction heads in the wild:** find and ablate them in GPT-2. ~2h
7. **Activation patching:** layer × position patching on IOI. ~2h
8. **IOI name movers:** head-level patching and resample ablation. ~3h
9. **Linear probes:** probe for "inside quotes" at every layer. ~2h
10. **Train a small SAE:** TopK or L1, 8× dictionary. GPU recommended. ~4h
11. **Pretrained SAE:** interpret a feature and steer with it. ~3h
12. **Capstone:** one week on an open question, 1-page writeup. ~1 week

## Workflow

Each exercise folder has `CHALLENGE.md`, a stubbed `solution.py` and `test_check.py`. Fill in the stub. Exploratory exercises (02–04, 06–11) write `results.json` when you run the script:

```sh
uv run python exercises/03-hooks-and-ablation/solution.py
uv run pytest exercises/03-*
```

01 and 05 test your functions directly, so you only need `pytest`. 12 has no check. Every failure says which condition failed and by how much.

Reference solutions are not in this repo.

## Layout

- `common/`: device choice, GPT-2 loading, datasets and prompts, results and test helpers.
- `exercises/NN-*/`: one folder per rung.
