from __future__ import annotations

import argparse
import math
import statistics
import timeit
from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import torch
from einops import einsum

import basics.model as _basics_model
from basics.model import BasicsTransformerLM
from basics.nn_utils import softmax as _softmax


@dataclass(frozen=True)
class ModelSpec:
    d_model: int
    d_ff: int
    num_layers: int
    num_heads: int


MODEL_SPECS: dict[str, ModelSpec] = {
    "small": ModelSpec(d_model=512, d_ff=2048, num_layers=8, num_heads=8),
    "medium": ModelSpec(d_model=768, d_ff=3072, num_layers=12, num_heads=12),
    "large": ModelSpec(d_model=1024, d_ff=4096, num_layers=24, num_heads=16),
}


@dataclass(frozen=True)
class BenchmarkConfig:
    model_size: str
    context_length: int = 128
    batch_size: int = 4
    vocab_size: int = 10_000
    warmup_steps: int = 5
    measure_steps: int = 10
    mode: Literal["forward", "forward-backward", "train-step"] = "forward"
    use_bf16: bool = False
    use_memory_profiler: bool = False
    compile_model: bool = False
    output_dir: Path = Path("artifacts")


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Benchmark and profile the Basics transformer.")
    parser.add_argument("--model-size", choices=sorted(MODEL_SPECS), required=True)
    parser.add_argument("--context-length", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--vocab-size", type=int, default=10_000)
    parser.add_argument("--warmup-steps", type=int, default=5)
    parser.add_argument("--measure-steps", type=int, default=10)
    parser.add_argument("--mode", choices=["forward", "forward-backward", "train-step"], default="forward")
    parser.add_argument("--use-bf16", action="store_true")
    parser.add_argument("--use-memory-profiler", action="store_true")
    parser.add_argument("--compile-model", action="store_true")
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts"))
    return parser


def build_model(config: BenchmarkConfig) -> torch.nn.Module:
    """Instantiate the staff Basics transformer for the requested model size."""
    spec = MODEL_SPECS[config.model_size]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = BasicsTransformerLM(
        vocab_size=config.vocab_size,
        context_length=config.context_length,
        d_model=spec.d_model,
        num_layers=spec.num_layers,
        num_heads=spec.num_heads,
        d_ff=spec.d_ff,
        rope_theta=10000.0,
    ).to(device)
    return model


def make_random_batch(config: BenchmarkConfig, device: torch.device) -> torch.Tensor:
    """Construct a random token batch for benchmarking and profiling."""
    return torch.randint(0, config.vocab_size, (config.batch_size, config.context_length), device=device)


def run_single_step(
    model: torch.nn.Module,
    batch: torch.Tensor,
    mode: Literal["forward", "forward-backward", "train-step"],
    autocast_context,
    optimizer: torch.optim.Optimizer | None = None,
) -> None:
    """Execute one benchmark step and synchronize CUDA before returning."""
    if mode != "forward":
        if optimizer is not None:
            optimizer.zero_grad()
        else:
            model.zero_grad()

    with autocast_context:
        if mode == "forward":
            with torch.no_grad():
                model(batch)
        else:
            logits = model(batch)
            logits.sum().backward()

    if mode == "train-step" and optimizer is not None:
        optimizer.step()

    torch.cuda.synchronize()


def benchmark_model(config: BenchmarkConfig) -> dict[str, float]:
    """Run warmup steps followed by timed measurement steps."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(config)
    model.train()

    if config.compile_model:
        model = torch.compile(model)

    batch = make_random_batch(config, device)
    autocast_ctx = make_autocast_context(config.use_bf16)

    optimizer = None
    if config.mode == "train-step":
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

    torch.cuda.reset_peak_memory_stats(device)

    times: list[float] = []
    for step in range(config.warmup_steps + config.measure_steps):
        is_warmup = step < config.warmup_steps

        if step == config.warmup_steps:
            maybe_start_memory_history(config.use_memory_profiler)

        torch.cuda.nvtx.range_push("warmup" if is_warmup else "measurement")
        t0 = timeit.default_timer()
        run_single_step(model, batch, config.mode, autocast_ctx, optimizer)
        t1 = timeit.default_timer()
        torch.cuda.nvtx.range_pop()
        if not is_warmup:
            times.append(t1 - t0)

    maybe_dump_memory_snapshot(
        config.use_memory_profiler,
        config.output_dir / "memory_snapshot.pickle",
    )

    peak_memory_mb = torch.cuda.max_memory_allocated(device) / (1024 ** 2)
    mean_s = statistics.mean(times)
    std_s = statistics.stdev(times) if len(times) > 1 else 0.0
    results = {
        "mean_s": mean_s,
        "std_s": std_s,
        "mean_ms": mean_s * 1000,
        "std_ms": std_s * 1000,
        "peak_memory_mb": peak_memory_mb,
    }
    print(
        f"[{config.model_size:6s} | {config.mode:16s}] "
        f"mean={mean_s*1000:7.2f} ms  std={std_s*1000:6.2f} ms"
        f"  peak_mem={peak_memory_mb:8.1f} MB"
    )
    return results


def annotated_scaled_dot_product_attention(Q, K, V, mask=None):
    """Drop-in replacement for scaled_dot_product_attention with NVTX range annotations."""
    nvtx = torch.cuda.nvtx
    nvtx.range_push("scaled dot product attention")

    nvtx.range_push("computing attention scores")
    d_k = K.shape[-1]
    scores = einsum(Q, K, "... query d_k, ... key d_k -> ... query key") / math.sqrt(d_k)
    if mask is not None:
        scores = torch.where(mask, scores, float("-inf"))
    nvtx.range_pop()

    nvtx.range_push("computing softmax")
    weights = _softmax(scores, dim=-1)
    nvtx.range_pop()

    nvtx.range_push("final matmul")
    out = einsum(weights, V, "... query key, ... key d_v -> ... query d_v")
    nvtx.range_pop()

    nvtx.range_pop()  # scaled dot product attention
    return out


# Patch at import time so every invocation — including nsys subprocesses — uses the
# annotated version. NVTX calls are no-ops when not running under nsys.
_basics_model.scaled_dot_product_attention = annotated_scaled_dot_product_attention


def maybe_start_memory_history(enabled: bool) -> None:
    if enabled:
        torch.cuda.memory._record_memory_history(max_entries=100_000)


def maybe_dump_memory_snapshot(enabled: bool, output_path: Path) -> None:
    if enabled:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        torch.cuda.memory._dump_snapshot(str(output_path))
        torch.cuda.memory._record_memory_history(enabled=None)


def make_autocast_context(use_bf16: bool):
    if use_bf16:
        return torch.autocast(device_type="cuda", dtype=torch.bfloat16)
    return nullcontext()


def main() -> None:
    args = build_argparser().parse_args()
    config = BenchmarkConfig(
        model_size=args.model_size,
        context_length=args.context_length,
        batch_size=args.batch_size,
        vocab_size=args.vocab_size,
        warmup_steps=args.warmup_steps,
        measure_steps=args.measure_steps,
        mode=args.mode,
        use_bf16=args.use_bf16,
        use_memory_profiler=args.use_memory_profiler,
        compile_model=args.compile_model,
        output_dir=args.output_dir,
    )
    benchmark_model(config)


if __name__ == "__main__":
    main()
