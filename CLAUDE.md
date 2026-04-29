# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

```sh
# Install dependencies
uv sync

# Run all public tests
uv run pytest -v ./tests

# Run a single test file
uv run pytest -v tests/test_alignment_utils.py
uv run pytest -v tests/test_grpo.py

# Run a single test by name
uv run pytest -v tests/test_alignment_utils.py::test_tokenize_prompt_and_output

# Run the transformer benchmark
uv run python -m systems.benchmark --model-size small --mode forward

# Run the attention benchmark
uv run python -m systems.attention_benchmark

# Create submission zip (runs tests first)
bash prepare_submission.sh
```

## Architecture

This homework has two independent parts:

### Part 2 — Systems / GPU Profiling (`systems/`)
Benchmarks a custom transformer from the `basics` package. All scaffold functions in `benchmark.py` and `attention_benchmark.py` are `raise NotImplementedError`. Key configurations:
- `ModelSpec` / `MODEL_SPECS`: small/medium/large transformer sizes
- `BenchmarkConfig`: controls batch size, context length, bf16, `torch.compile`, memory profiling, NVTX annotation
- `AttentionBenchmarkConfig`: sweeps over head dims × sequence lengths for Section 2.7–2.8

### Part 3 — Alignment / GRPO (`alignment/`)
Fine-tunes `Qwen/Qwen2.5-Math-1.5B` on GSM8K using GRPO. All scaffold functions are `raise NotImplementedError`. Flow:
- **`eval.py`**: loads GSM8K from HuggingFace, runs direct/CoT/self-consistency baselines via `vllm`
- **`grpo.py`**: core GRPO helpers (`tokenize_prompt_and_output`, `get_response_log_probs`, `compute_group_normalized_rewards`, `compute_grpo_clip_loss`, `grpo_microbatch_train_step`) and the full training loop
- **`rewards.py`**: provided grading utilities — expects model outputs to use `<answer>...</answer>` tags; delegates math equivalence checking to `drgrpo_grader.py`
- **`prompts.py`**: `DIRECT_PROMPT_TEMPLATE` (inline string) and `COT_PROMPT_TEMPLATE` (lazy-loaded from `prompt.txt` at repo root)
- **`drgrpo_grader.py`**: third-party math grader (Apache 2.0); do not modify

### `basics/` — Shared Transformer Package
Local editable install (`basics/basics/`). Implements `BasicsTransformerLM` (pre-norm, RoPE, SwiGLU, RMSNorm). Used by `systems/benchmark.py` to instantiate models. The `basics` package is a dependency from a prior homework assignment.

### Tests
`tests/adapters.py` is the indirection layer: all tests import from adapters, which re-export from `alignment.grpo`. This allows the test suite to be run without modifying test files. `tests/conftest.py` provides lightweight toy fixtures (`ToyTokenizer`, `ToyCausalLM`) — no pretrained model needed.

### Notes
- `vllm` is **not** installed by `uv sync`; it is GPU-only and installed separately inside the Colab notebook (`colab_setup.ipynb`) for Sections 3.1–3.2 evaluation.
- The `prepare_submission.sh` script zips everything except `.venv`, build artifacts, model weights, and most non-source files. It does **not** exclude `.ipynb` files or `.md` files.
