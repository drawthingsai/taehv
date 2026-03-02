#!/usr/bin/env python3
"""Compare first-frame Wan 2.1 CoreML latency before/after."""
import argparse
import re
import subprocess
from pathlib import Path


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--before-model", required=True, help="Path to baseline .mlmodelc")
    p.add_argument("--after-model", required=True, help="Path to candidate .mlmodelc")
    p.add_argument("--iters", type=int, default=200)
    p.add_argument("--warmup", type=int, default=20)
    p.add_argument("--compute-units", default="cpu_ne", choices=["cpu_only", "cpu_ne", "cpu_gpu", "all"])
    p.add_argument("--dtype", default="float16", choices=["float16", "float32"])
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--swift-source", default="scripts/benchmark_wan21_firstframe.swift")
    p.add_argument("--swift-bin", default="/tmp/benchmark_wan21_firstframe")
    return p.parse_args()


def compile_swift(swift_source: str, swift_bin: str) -> None:
    cmd = ["swiftc", "-O", swift_source, "-o", swift_bin]
    print("Compiling benchmark binary:", " ".join(cmd))
    subprocess.run(cmd, check=True)


def run_one(
    model_path: str,
    swift_bin: str,
    iters: int,
    warmup: int,
    compute_units: str,
    dtype: str,
    seed: int,
) -> tuple[float, float, str]:
    cmd = [
        swift_bin,
        model_path,
        "--iters", str(iters),
        "--warmup", str(warmup),
        "--compute-units", compute_units,
        "--dtype", dtype,
        "--seed", str(seed),
    ]
    proc = subprocess.run(cmd, check=True, capture_output=True, text=True)
    out = proc.stdout.strip()
    m = re.search(r"Latency ms: mean=([0-9.]+).*median=([0-9.]+)", out)
    if m is None:
        raise RuntimeError(f"Could not parse benchmark output for {model_path}\n{out}")
    mean = float(m.group(1))
    median = float(m.group(2))
    return mean, median, out


def main() -> None:
    args = parse_args()
    swift_source = Path(args.swift_source)
    swift_bin = Path(args.swift_bin)
    if not swift_source.exists():
        raise SystemExit(f"Swift source not found: {swift_source}")
    compile_swift(str(swift_source), str(swift_bin))

    print("\nRunning BEFORE model benchmark...")
    before_mean, before_median, before_out = run_one(
        model_path=args.before_model,
        swift_bin=str(swift_bin),
        iters=args.iters,
        warmup=args.warmup,
        compute_units=args.compute_units,
        dtype=args.dtype,
        seed=args.seed,
    )
    print(before_out)

    print("\nRunning AFTER model benchmark...")
    after_mean, after_median, after_out = run_one(
        model_path=args.after_model,
        swift_bin=str(swift_bin),
        iters=args.iters,
        warmup=args.warmup,
        compute_units=args.compute_units,
        dtype=args.dtype,
        seed=args.seed,
    )
    print(after_out)

    delta_mean = after_mean - before_mean
    delta_median = after_median - before_median
    pct_mean = (delta_mean / before_mean * 100.0) if before_mean != 0 else 0.0
    pct_median = (delta_median / before_median * 100.0) if before_median != 0 else 0.0

    print("\nComparison Summary")
    print(f"Before mean: {before_mean:.3f} ms")
    print(f"After mean:  {after_mean:.3f} ms")
    print(f"Delta mean:  {delta_mean:+.3f} ms ({pct_mean:+.2f}%)")
    print(f"Before median: {before_median:.3f} ms")
    print(f"After median:  {after_median:.3f} ms")
    print(f"Delta median:  {delta_median:+.3f} ms ({pct_median:+.2f}%)")


if __name__ == "__main__":
    main()
