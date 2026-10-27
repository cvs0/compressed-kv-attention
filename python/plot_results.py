#!/usr/bin/env python3
"""Charts from kvexp_bench JSON and the exponent-entropy reports.

Reads only numbers stored in those files. Writes PNG figures.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def load(path: Path) -> dict:
    return json.loads(path.read_text())


def plot_ratios(entropy_paths: list[Path], out: Path) -> None:
    fig, ax = plt.subplots(figsize=(8.2, 4.2))
    for path in entropy_paths:
        doc = load(path)
        seq = doc["analyzed_seq_len"]
        layers = doc["layers_at_best_block"]
        xs = [row["layer"] for row in layers]
        ax.plot(xs, [row["K"]["ratio"] for row in layers], marker="o", ms=3, label=f"K {seq}")
        ax.plot(xs, [row["V"]["ratio"] for row in layers], marker="s", ms=3, label=f"V {seq}")
    ax.set_xlabel("layer")
    ax.set_ylabel("container bytes / tight BF16 bytes")
    ax.set_title("XKV ratio by layer, block 128, modes {raw, 2-bit, 4-bit}")
    ax.set_ylim(0.70, 0.78)
    ax.grid(True, alpha=0.3)
    ax.legend(ncol=3, fontsize=8)
    fig.tight_layout()
    fig.savefig(out, dpi=140)
    plt.close(fig)


def plot_latency(bench: dict, out: Path) -> None:
    runs = [r for r in bench["runs"] if r["cold"] == 1 and r["backend"] == "avx512bf16"]
    if not runs:
        runs = [r for r in bench["runs"] if r["cold"] == 1]
    scopes = ["head", "model", "model_heads"]
    fig, axes = plt.subplots(1, len(scopes), figsize=(11.5, 3.8), sharex=True)
    for ax, scope in zip(axes, scopes):
        subset = [r for r in runs if r["scope"] == scope]
        threads = sorted({r["threads"] for r in subset})
        for th in threads:
            rows = sorted((r for r in subset if r["threads"] == th), key=lambda r: r["seq"])
            xs = [r["seq"] for r in rows]
            ax.plot(xs, [r["raw_ms"] for r in rows], marker="o", label=f"raw t={th}")
            ax.plot(xs, [r["xkv_ms"] for r in rows], marker="x", linestyle="--", label=f"xkv t={th}")
        ax.set_title(scope)
        ax.set_xlabel("context")
        ax.set_ylabel("median ms")
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=7, ncol=2)
    backend = runs[0]["backend"] if runs else "none"
    fig.suptitle(f"cold decode step, {backend}")
    fig.tight_layout()
    fig.savefig(out, dpi=140)
    plt.close(fig)


def plot_bandwidth(bench: dict, out: Path) -> None:
    fig, ax = plt.subplots(figsize=(7.4, 4.2))
    stream = [s for s in bench["stream_read"] if s.get("method") == "avx512_load" and s.get("threads", 1) == 1]
    if not stream:
        stream = bench["stream_read"]
    ax.plot([s["bytes"] / (1024**2) for s in stream], [s["gbps"] for s in stream], marker="o", color="black", label="avx512 load, 1 thread")
    runs = [r for r in bench["runs"] if r["cold"] == 1 and r["scope"] == "model_heads" and r["threads"] == 4]
    for backend in sorted({r["backend"] for r in runs}):
        rows = sorted((r for r in runs if r["backend"] == backend), key=lambda r: r["seq"])
        xs = [r["tight_bytes"] / (1024**2) for r in rows]
        ax.plot(xs, [r["raw_gbps_tight"] for r in rows], marker="o", label=f"{backend} raw")
        ax.plot(xs, [r["xkv_gbps_container"] for r in rows], marker="x", linestyle="--", label=f"{backend} xkv")
    ax.set_xlabel("bytes touched (MiB)")
    ax.set_ylabel("GB/s (1e9)")
    ax.set_title("cold bandwidth, model_heads, 4 threads")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out, dpi=140)
    plt.close(fig)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--bench", type=Path, required=True)
    p.add_argument("--entropy", type=Path, nargs="+", required=True)
    p.add_argument("--outdir", type=Path, required=True)
    args = p.parse_args()
    args.outdir.mkdir(parents=True, exist_ok=True)
    bench = load(args.bench)
    plot_ratios(list(args.entropy), args.outdir / "ratio_by_layer.png")
    plot_latency(bench, args.outdir / "latency_vs_context.png")
    plot_bandwidth(bench, args.outdir / "bandwidth.png")
    print(f"wrote {args.outdir}")


if __name__ == "__main__":
    main()
