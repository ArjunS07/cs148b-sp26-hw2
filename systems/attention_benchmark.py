from __future__ import annotations

import argparse
import statistics
import timeit
from dataclasses import dataclass
from typing import Iterable

import torch

from basics.model import scaled_dot_product_attention as _sdpa

# Replaced by benchmark_attention_grid when compile_attention=True.
_attn_fn = _sdpa

_WARMUP_STEPS = 5


@dataclass(frozen=True)
class AttentionBenchmarkConfig:
    head_dims: tuple[int, ...] = (16, 32, 64, 128)
    sequence_lengths: tuple[int, ...] = (64, 128, 256, 512, 1024)
    batch_size: int = 8
    forward_passes: int = 100
    backward_passes: int = 100
    compile_attention: bool = False


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Benchmark attention implementations.")
    parser.add_argument("--compile-attention", action="store_true")
    return parser


def iter_benchmark_shapes(config: AttentionBenchmarkConfig) -> Iterable[tuple[int, int]]:
    for head_dim in config.head_dims:
        for sequence_length in config.sequence_lengths:
            yield head_dim, sequence_length


def make_qkv(batch_size: int, sequence_length: int, head_dim: int, device: torch.device) -> tuple[torch.Tensor, ...]:
    """Create random Q, K, V with shape (batch, 1 head, seq_len, head_dim)."""
    shape = (batch_size, 1, sequence_length, head_dim)
    q = torch.randn(shape, device=device, requires_grad=True)
    k = torch.randn(shape, device=device, requires_grad=True)
    v = torch.randn(shape, device=device, requires_grad=True)
    return q, k, v


def benchmark_attention_once(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> dict[str, float]:
    """Time the forward and backward pass for a single attention configuration."""
    # Warmup: full forward+backward to JIT-compile kernels and fill caches.
    for _ in range(_WARMUP_STEPS):
        out = _attn_fn(q, k, v)
        out.sum().backward()
        torch.cuda.synchronize()
        q.grad = k.grad = v.grad = None

    # Time forward passes with no_grad (measures compute only, not graph construction).
    fwd_times: list[float] = []
    for _ in range(100):
        t0 = timeit.default_timer()
        with torch.no_grad():
            _attn_fn(q, k, v)
        torch.cuda.synchronize()
        fwd_times.append(timeit.default_timer() - t0)

    # Measure activation memory saved for the backward pass: run one forward and read
    # memory_allocated() before calling backward — that includes the cached intermediates.
    out = _attn_fn(q, k, v)
    torch.cuda.synchronize()
    mem_before_bwd_bytes = torch.cuda.memory_allocated()

    # Time backward passes; each needs its own fresh forward to rebuild the graph.
    bwd_times: list[float] = []
    for _ in range(100):
        out = _attn_fn(q, k, v)
        torch.cuda.synchronize()
        t0 = timeit.default_timer()
        out.sum().backward()
        torch.cuda.synchronize()
        bwd_times.append(timeit.default_timer() - t0)
        q.grad = k.grad = v.grad = None

    return {
        "fwd_mean_ms": statistics.mean(fwd_times) * 1000,
        "fwd_std_ms": statistics.stdev(fwd_times) * 1000,
        "bwd_mean_ms": statistics.mean(bwd_times) * 1000,
        "bwd_std_ms": statistics.stdev(bwd_times) * 1000,
        "mem_before_bwd_mb": mem_before_bwd_bytes / (1024 ** 2),
    }


def benchmark_attention_grid(config: AttentionBenchmarkConfig) -> list[dict[str, float | int | str]]:
    """Run the attention benchmark over the Section 2.7 Cartesian product of scales."""
    global _attn_fn
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if config.compile_attention:
        _attn_fn = torch.compile(_sdpa)

    rows: list[dict] = []
    for head_dim, seq_len in iter_benchmark_shapes(config):
        print(f"head_dim={head_dim:4d}  seq_len={seq_len:5d} ...", end=" ", flush=True)
        try:
            q, k, v = make_qkv(config.batch_size, seq_len, head_dim, device)
            r = benchmark_attention_once(q, k, v)
            r.update({"head_dim": head_dim, "seq_len": seq_len})
            rows.append(r)
            print(
                f"fwd={r['fwd_mean_ms']:7.3f} ms  "
                f"bwd={r['bwd_mean_ms']:7.3f} ms  "
                f"mem={r['mem_before_bwd_mb']:8.2f} MB"
            )
        except torch.cuda.OutOfMemoryError:
            rows.append({"head_dim": head_dim, "seq_len": seq_len, "fwd_mean_ms": "OOM", "bwd_mean_ms": "OOM", "mem_before_bwd_mb": "OOM"})
            print("OOM")
            torch.cuda.empty_cache()

    _print_summary(rows)
    return rows


def _print_summary(rows: list[dict]) -> None:
    header = f"{'head_dim':>8}  {'seq_len':>7}  {'fwd (ms)':>10}  {'bwd (ms)':>10}  {'mem (MB)':>10}"
    sep = "-" * len(header)
    print(f"\n=== Attention Benchmark Results ===\n{header}\n{sep}")
    for r in rows:
        fwd = r["fwd_mean_ms"] if isinstance(r["fwd_mean_ms"], str) else f"{r['fwd_mean_ms']:.3f}"
        bwd = r["bwd_mean_ms"] if isinstance(r["bwd_mean_ms"], str) else f"{r['bwd_mean_ms']:.3f}"
        mem = r["mem_before_bwd_mb"] if isinstance(r["mem_before_bwd_mb"], str) else f"{r['mem_before_bwd_mb']:.2f}"
        print(f"{r['head_dim']:>8}  {r['seq_len']:>7}  {fwd:>10}  {bwd:>10}  {mem:>10}")


def main() -> None:
    args = build_argparser().parse_args()
    config = AttentionBenchmarkConfig(compile_attention=args.compile_attention)
    benchmark_attention_grid(config)


if __name__ == "__main__":
    main()
