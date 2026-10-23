# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

A learning repo: twelve mech-interp exercises on GPT-2 small and toy models, done in order. The user is the learner. Philosophy is "build first": each `CHALLENGE.md` gives a goal, a check and blind spots, not a tutorial. Explain concepts and blind spots when asked; don't write the learner's `solution.py` unless explicitly asked.

Reference solutions live outside this repo at `../reference-solutions/NN-*/solution.py` (deliberately not committed). Don't reveal them unless asked.

## Directory map

```
../reference-solutions/NN-*/solution.py   # outside repo, 01–11 only
mech-interp-ladder/
  common/          # models, device, data, results, checks (see below)
  exercises/
    conftest.py
    01-transformer-from-scratch/   # function tests
    02-toy-attention-only/
    03-hooks-and-ablation/
    04-logit-lens/
    05-direct-logit-attribution/   # function tests
    06-induction-heads/
    07-activation-patching/
    08-ioi-name-movers/
    09-linear-probes/
    10-train-an-sae/
    11-pretrained-sae/
    12-capstone/                   # CHALLENGE.md + WRITEUP.md, no check
```

Exercises 01–11 hold only `CHALLENGE.md`, `solution.py`, `test_check.py` (plus a gitignored `results.json` once run).

## Commands

```sh
uv sync                                                    # install (Python 3.11–3.12)
uv run python exercises/03-hooks-and-ablation/solution.py  # run an exploratory exercise, writes results.json
uv run pytest exercises/03-*                               # check one exercise
uv run pytest exercises/03-*/test_check.py::test_some_head_matters  # single test
```

No lint or build step.

## How exercises are checked

Each `exercises/NN-*/` has `CHALLENGE.md`, a stubbed `solution.py` (functions raise `NotImplementedError`) and `test_check.py`. Two check styles:

- **Function tests (01, 05):** tests import `solution.py` via `common.checks.load_solution` and call functions through `call()`, which turns `NotImplementedError` into a clear failure. Just run `pytest`.
- **Results tests (02–04, 06–11):** running `solution.py` saves `results.json` next to itself (`common.results.save_results(__file__, ...)`); tests read it via `load_results`, which fails clearly if missing. Run the script before `pytest`. `results.json` is gitignored.
- 12 (capstone) has no check.

`exercises/conftest.py` reports deliberate `pytest.fail` in fixtures as FAILED rather than ERROR. Pytest uses `--import-mode=importlib`, so every folder's `solution.py`/`test_check.py` can share names.

## Shared code (`common/`)

- `models.load_gpt2(device, processing=True)`: `processing=False` uses `from_pretrained_no_processing` (raw HF weights). Exercise 01 needs unprocessed weights so a from-scratch model can `load_state_dict(..., strict=False)` and match logits.
- `device.get_device()`: CUDA → MPS → CPU.
- `data`: pile-10k loader, `FACTUAL_PROMPTS`, IOI prompt pairs (`get_ioi_pairs`, one template so positions align), synthetic quote texts for probes.
- `checks.head_keys()`: canonical head names `"L{layer}H{head}"` used as results keys across exercises. `KNOWN_NAME_MOVERS` holds the IOI paper's name movers.

Stubs use `jaxtyping` shape annotations (`Float[Tensor, "batch pos d_model"]`); keep them when editing.
