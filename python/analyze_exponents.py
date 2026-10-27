#!/usr/bin/env python3
"""Exponent entropy and fixed-rate codec size on a KVD1 dump.

The codec measured here is the one the attention kernel consumes: per block of
tokens, a per-dimension base exponent, a fixed-width residual with a reserved
all-ones escape sentinel, and a raw BF16 escape stream in scan order. Blocks
independently choose raw BF16, 2-bit, or 4-bit residuals by stored size.
"""

from __future__ import annotations

import argparse
import json
import math
import zlib
from pathlib import Path

import numpy as np

MAGIC = np.frombuffer(b"KVD1", dtype="<u4")[0]


def load_dump(path: Path):
    with path.open("rb") as f:
        header = np.frombuffer(f.read(32), dtype="<u4")
        if int(header[0]) != int(MAGIC) or int(header[1]) != 1:
            raise SystemExit("not a KVD1 version-1 dump")
        n_layers, n_kv, n_q, dim, seq = map(int, header[2:7])
        count = n_layers * n_kv * seq * dim
        k = np.frombuffer(f.read(count * 2), dtype="<u2").copy().reshape(n_layers, n_kv, seq, dim)
        v = np.frombuffer(f.read(count * 2), dtype="<u2").copy().reshape(n_layers, n_kv, seq, dim)
    return k, v, {"n_layers": n_layers, "n_kv_heads": n_kv, "n_q_heads": n_q, "head_dim": dim, "seq_len": seq}


def shannon_from_counts(counts: np.ndarray) -> float:
    total = float(counts.sum())
    if total <= 0:
        return 0.0
    p = counts.astype(np.float64)
    p = p[p > 0] / total
    return float(-(p * np.log2(p)).sum())


def shannon_u8(values: np.ndarray) -> float:
    return shannon_from_counts(np.bincount(values.reshape(-1), minlength=256))


def conditional_dim_entropy(field: np.ndarray) -> float:
    # field: [seq, dim], values in 0..255. Equal weight per dimension.
    seq, dim = field.shape
    ent = 0.0
    for d in range(dim):
        ent += shannon_from_counts(np.bincount(field[:, d], minlength=256))
    return ent / dim


def block_conditional_entropy(exp: np.ndarray, block: int) -> float:
    # exp: [seq, dim]. Mean Shannon entropy of the exponent inside each (block, dim).
    seq, dim = exp.shape
    nblk = seq // block
    if nblk == 0:
        return float("nan")
    ee = exp[: nblk * block].reshape(nblk, block, dim).transpose(0, 2, 1)
    groups, b = ee.shape[0] * ee.shape[1], block
    flat = ee.reshape(groups, b)
    idx = (np.arange(groups, dtype=np.int64)[:, None] * 256 + flat.astype(np.int64)).ravel()
    hist = np.bincount(idx, minlength=groups * 256).reshape(groups, 256).astype(np.float64)
    p = hist / b
    with np.errstate(divide="ignore", invalid="ignore"):
        logp = np.where(hist > 0, np.log2(p), 0.0)
    h = -(p * logp).sum(axis=1)
    return float(h.mean())


def best_window_escapes(exp: np.ndarray, block: int, width: int) -> tuple[int, int]:
    """Return (n_elements, n_escapes) for non-overlapping blocks.

    The tail that does not fill a block is included as its own block.
    `width` is the number of consecutive exponents covered by non-sentinel codes.
    """
    seq, dim = exp.shape
    escapes = 0
    covered = 0
    for start in range(0, seq, block):
        piece = exp[start : start + block]
        n = piece.shape[0]
        covered += n * dim
        # hist per dim: [dim, 256]
        idx = (np.arange(dim, dtype=np.int64)[None, :] * 256 + piece.astype(np.int64)).ravel()
        # piece is [n, dim], want group = dim index
        idx = (np.arange(dim, dtype=np.int64)[None, :] * 256 + piece.astype(np.int64)).ravel()
        hist = np.bincount(idx, minlength=dim * 256).reshape(dim, 256)
        csum = np.cumsum(hist, axis=1)
        best = np.zeros(dim, dtype=np.int64)
        last_base = 256 - width
        for base in range(0, last_base + 1):
            end = base + width - 1
            cnt = csum[:, end] if base == 0 else csum[:, end] - csum[:, base - 1]
            best = np.maximum(best, cnt)
        escapes += int(n * dim - best.sum())
    return covered, escapes


def coded_bytes(n_tokens: int, dim: int, bits: int, n_escape: int) -> int:
    exp_bytes = (n_tokens * dim * bits + 7) // 8
    return 8 + dim + n_tokens * dim + exp_bytes + 2 * n_escape


def raw_container_bytes(n_tokens: int, dim: int) -> int:
    return 8 + 2 * n_tokens * dim


def tight_bytes(n_tokens: int, dim: int) -> int:
    return 2 * n_tokens * dim


def policy_size(exp: np.ndarray, block: int, bit_widths: tuple[int, ...]) -> dict:
    """Per-block min of raw container and sentinel fixed-rate codes."""
    seq, dim = exp.shape
    total_container = 0
    total_tight = tight_bytes(seq, dim)
    total_esc = 0
    total_el = seq * dim
    coded_el = 0
    mode_blocks = {0: 0, 2: 0, 3: 0, 4: 0}
    for start in range(0, seq, block):
        piece = exp[start : start + block]
        n = piece.shape[0]
        options = [(0, raw_container_bytes(n, dim), 0)]
        idx = (np.arange(dim, dtype=np.int64)[None, :] * 256 + piece.astype(np.int64)).ravel()
        hist = np.bincount(idx, minlength=dim * 256).reshape(dim, 256)
        csum = np.cumsum(hist, axis=1, dtype=np.int64)
        csum_pad = np.concatenate([np.zeros((dim, 1), dtype=np.int64), csum], axis=1)
        for bits in bit_widths:
            width = (1 << bits) - 1
            nbase = 256 - width + 1
            bases = np.arange(nbase)
            cnt = csum[:, bases + width - 1] - csum_pad[:, bases]
            best = cnt.max(axis=1)
            n_esc = int(n * dim - int(best.sum()))
            options.append((bits, coded_bytes(n, dim, bits, n_esc), n_esc))
        # Smaller container, then raw, then fewer escapes, then narrower codes.
        mode, size, n_esc = min(options, key=lambda t: (t[1], 0 if t[0] == 0 else 1, t[2], t[0]))
        total_container += size
        total_esc += n_esc
        mode_blocks[mode] = mode_blocks.get(mode, 0) + 1
        if mode != 0:
            coded_el += n * dim
    return {
        "container_bytes": int(total_container),
        "tight_bytes": int(total_tight),
        "ratio": total_container / total_tight,
        "escape_rate": total_esc / total_el,
        "coded_element_fraction": coded_el / total_el,
        "mode_blocks": {str(k): int(v) for k, v in mode_blocks.items() if v},
    }


def field_report(bits: np.ndarray, block_for_entropy: int) -> dict:
    # bits: [heads, seq, dim] uint16
    exp = ((bits >> 7) & 0xFF).astype(np.uint8)
    sign = ((bits >> 15) & 0x1).astype(np.uint8)
    mant = (bits & 0x7F).astype(np.uint8)
    heads = exp.shape[0]
    # Pool heads for global histograms; also summarize per head.
    per_head = []
    for h in range(heads):
        e = exp[h].reshape(-1)
        per_head.append(
            {
                "H_exp": shannon_u8(e),
                "H_sign": shannon_from_counts(np.bincount(sign[h].reshape(-1), minlength=2)),
                "H_mant": shannon_u8(mant[h].reshape(-1)),
                "H_exp_given_dim": conditional_dim_entropy(exp[h]),
                "zero_frac": float(((bits[h] & 0x7FFF) == 0).mean()),
                "unique_exp": int(np.unique(e).size),
            }
        )
    flat_exp = exp.reshape(-1)
    return {
        "H_exp": shannon_u8(flat_exp),
        "H_sign": shannon_from_counts(np.bincount(sign.reshape(-1), minlength=2)),
        "H_mant": shannon_u8(mant.reshape(-1)),
        "H_exp_given_dim": float(np.mean([h["H_exp_given_dim"] for h in per_head])),
        "H_exp_given_block_dim": float(
            np.mean([block_conditional_entropy(exp[h], block_for_entropy) for h in range(heads)])
        ),
        "zero_frac": float(((bits & 0x7FFF) == 0).mean()),
        "subnormal_frac": float(((bits & 0x7F80) == 0).mean() - ((bits & 0x7FFF) == 0).mean()),
        "per_head": per_head,
        "exp": exp,
    }


def zlib_ratio(sample: np.ndarray) -> dict:
    raw = np.ascontiguousarray(sample).tobytes()
    out = {}
    for level in (1, 6):
        comp = zlib.compress(raw, level)
        out[f"level_{level}"] = len(comp) / len(raw)
    return out


def analyze(k: np.ndarray, v: np.ndarray, block_sizes: list[int], bit_widths: tuple[int, ...]) -> dict:
    # k, v: [layers, heads, seq, dim]
    n_layers = k.shape[0]
    sweep = []
    for block in block_sizes:
        k_bytes = v_bytes = k_tight = v_tight = 0
        k_esc_num = v_esc_num = 0
        k_el = v_el = 0
        for layer in range(n_layers):
            for name, tensor, acc in (
                ("K", k[layer], "k"),
                ("V", v[layer], "v"),
            ):
                for h in range(tensor.shape[0]):
                    exp = ((tensor[h] >> 7) & 0xFF).astype(np.uint8)
                    stats = policy_size(exp, block, bit_widths)
                    if name == "K":
                        k_bytes += stats["container_bytes"]
                        k_tight += stats["tight_bytes"]
                        k_esc_num += stats["escape_rate"] * exp.size
                        k_el += exp.size
                    else:
                        v_bytes += stats["container_bytes"]
                        v_tight += stats["tight_bytes"]
                        v_esc_num += stats["escape_rate"] * exp.size
                        v_el += exp.size
        total_c = k_bytes + v_bytes
        total_t = k_tight + v_tight
        sweep.append(
            {
                "block": block,
                "bits": list(bit_widths),
                "K_ratio": k_bytes / k_tight,
                "V_ratio": v_bytes / v_tight,
                "KV_ratio": total_c / total_t,
                "K_escape_rate": k_esc_num / k_el,
                "V_escape_rate": v_esc_num / v_el,
                "KV_escape_rate": (k_esc_num + v_esc_num) / (k_el + v_el),
                "container_bytes": int(total_c),
                "tight_bytes": int(total_t),
            }
        )
    best = min(sweep, key=lambda s: (s["KV_ratio"], s["KV_escape_rate"], s["block"]))
    # Per-layer detail at the chosen block size, bits {2, 4} (kernel modes).
    kernel_bits = (2, 4)
    layers = []
    for layer in range(n_layers):
        row = {"layer": layer}
        for name, tensor in (("K", k[layer]), ("V", v[layer])):
            rep = field_report(tensor, best["block"])
            sizes = [
                policy_size(((tensor[h] >> 7) & 0xFF).astype(np.uint8), best["block"], kernel_bits)
                for h in range(tensor.shape[0])
            ]
            container = sum(s["container_bytes"] for s in sizes)
            tight = sum(s["tight_bytes"] for s in sizes)
            esc = sum(s["escape_rate"] * s["tight_bytes"] // 2 for s in sizes)
            elements = sum(tensor[h].size for h in range(tensor.shape[0]))
            row[name] = {
                "H_exp": rep["H_exp"],
                "H_sign": rep["H_sign"],
                "H_mant": rep["H_mant"],
                "H_exp_given_dim": rep["H_exp_given_dim"],
                "H_exp_given_block_dim": rep["H_exp_given_block_dim"],
                "zero_frac": rep["zero_frac"],
                "subnormal_frac": rep["subnormal_frac"],
                "ratio": container / tight,
                "escape_rate": esc / elements,
                "per_head_H_exp": [h["H_exp"] for h in rep["per_head"]],
                "per_head_H_exp_given_dim": [h["H_exp_given_dim"] for h in rep["per_head"]],
            }
        layers.append(row)
    # Theoretical savings if exponents were coded at H(exp | block, dim) and
    # sign+mantissa stayed at 8 bits: (8 - H) / 16, ignoring headers.
    h_cond = float(
        np.mean([(row["K"]["H_exp_given_block_dim"] + row["V"]["H_exp_given_block_dim"]) / 2 for row in layers])
    )
    theoretical_ratio = (8.0 + h_cond) / 16.0
    decision_ratio = best["KV_ratio"]
    decision = "GO" if decision_ratio <= 0.85 else "NO-GO"
    zlib_layer0 = {
        "K": zlib_ratio(k[0]),
        "V": zlib_ratio(v[0]),
    }
    return {
        "criteria": {
            "rule": "GO if the per-block min(raw, 2-bit, 4-bit) container is at most 85% of tight BF16 KV bytes",
            "threshold_ratio": 0.85,
            "bit_widths": [2, 4],
            "escape": "all-ones sentinel, per-dimension contiguous window, optimal base, smallest base on ties",
            "why_0.85": "Below a 15% byte cut the nibble decode in the fused loop is unlikely to repay itself on a bandwidth roofline.",
        },
        "decision": decision,
        "best_kernel_policy": best,
        "theoretical_ratio_at_block_conditional_entropy": theoretical_ratio,
        "mean_H_exp_given_block_dim": h_cond,
        "sweep_including_extra_widths": sweep,
        "layers_at_best_block": layers,
        "zlib_layer0_ratio": zlib_layer0,
    }


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--dump", type=Path, required=True)
    p.add_argument("--meta", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--prefix", type=int, default=0, help="If >0, analyze only the first N tokens")
    p.add_argument("--blocks", type=int, nargs="+", default=[16, 32, 64, 128])
    args = p.parse_args()
    k, v, info = load_dump(args.dump)
    if args.prefix:
        if args.prefix > info["seq_len"]:
            raise SystemExit("prefix longer than dump")
        k = k[:, :, : args.prefix, :]
        v = v[:, :, : args.prefix, :]
        info = dict(info)
        info["seq_len"] = args.prefix
        info["prefix_of"] = int(np.frombuffer(args.dump.read_bytes()[:32], dtype="<u4")[6]) if False else info.get("seq_len")
    meta = json.loads(args.meta.read_text())
    # Kernel policy sweep is 2-bit and 4-bit. A 3-bit sweep is reported separately.
    kernel = analyze(k, v, args.blocks, (2, 4))
    extra = []
    for block in args.blocks:
        # Reuse analyze's sweep by calling policy over one width set. Cheap enough
        # only for the winning block: compute 3-bit and {2,3,4} at every block via analyze.
        pass
    three = analyze(k, v, args.blocks, (2, 3, 4))
    only3 = analyze(k, v, [kernel["best_kernel_policy"]["block"]], (3,))
    kernel["sweep_2_3_4"] = three["sweep_including_extra_widths"]
    kernel["only_3bit_at_best_block"] = only3["sweep_including_extra_widths"]
    payload = {
        "source": meta,
        "analyzed_seq_len": info["seq_len"],
        "n_layers": info["n_layers"],
        "n_kv_heads": info["n_kv_heads"],
        "head_dim": info["head_dim"],
        **kernel,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2) + "\n")
    best = kernel["best_kernel_policy"]
    print(
        f"seq={info['seq_len']} decision={kernel['decision']} "
        f"block={best['block']} KV_ratio={best['KV_ratio']:.4f} "
        f"K={best['K_ratio']:.4f} V={best['V_ratio']:.4f} "
        f"escape={best['KV_escape_rate']:.4e} "
        f"H(exp|block,dim)={kernel['mean_H_exp_given_block_dim']:.3f} "
        f"theoretical_ratio={kernel['theoretical_ratio_at_block_conditional_entropy']:.4f}"
    )
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
