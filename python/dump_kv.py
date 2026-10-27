#!/usr/bin/env python3
"""Dump post-RoPE BF16 K and V from Qwen2.5-0.5B on a real document.

Prefill is chunked so the score tensor stays within a byte budget. K and V are
the tensors the model writes into its cache (after rotary position embedding).
The dump is append-only along the sequence, so a prefix is the cache that
existed after that many tokens.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.models.qwen2.modeling_qwen2 import apply_rotary_pos_emb

MAGIC = b"KVD1"
VERSION = 1


def to_u16(t: torch.Tensor) -> np.ndarray:
    t = t.detach().to(dtype=torch.bfloat16).contiguous().cpu()
    return t.view(torch.int16).numpy().view(np.uint16).copy()


def choose_chunk(past_len: int, remaining: int, n_q_heads: int, budget_bytes: int) -> int:
    # fp32 score tensor is [q_heads, chunk, past_len + chunk].
    cap = max(1, budget_bytes // (4 * n_q_heads))
    disc = past_len * past_len + 4 * cap
    chunk = int((-past_len + math.sqrt(disc)) / 2)
    chunk = max(1, min(remaining, chunk, 4096))
    return chunk


def install_q_hooks(model, captured: list, capture_flag: dict) -> None:
    for i, layer in enumerate(model.model.layers):
        attn = layer.self_attn
        orig = attn.forward

        def wrapped(
            hidden_states,
            position_embeddings,
            attention_mask,
            past_key_values=None,
            _attn=attn,
            _i=i,
            _orig=orig,
            **kwargs,
        ):
            if capture_flag["on"]:
                hs = hidden_states[:, -1:, :]
                q = _attn.q_proj(hs)
                bsz, q_len, _ = hs.shape
                q = q.view(bsz, q_len, -1, _attn.head_dim).transpose(1, 2)
                cos, sin = position_embeddings
                q_rot, _ = apply_rotary_pos_emb(q, q, cos[:, -1:, :], sin[:, -1:, :])
                captured[_i] = (
                    q_rot[0, :, 0, :].detach().to(torch.bfloat16).contiguous().cpu()
                )
            return _orig(
                hidden_states=hidden_states,
                position_embeddings=position_embeddings,
                attention_mask=attention_mask,
                past_key_values=past_key_values,
                **kwargs,
            )

        attn.forward = wrapped


def prefill(
    model,
    input_ids: torch.Tensor,
    budget_bytes: int,
    capture_flag: dict,
    captured: list | None = None,
    q_stops: list[int] | None = None,
):
    """Chunked prefill.

    If `q_stops` is set, chunk edges are forced onto those token counts and the
    post-RoPE queries at the last token of each stop are copied into `q_snaps`,
    keyed by the stop (the query that attends to exactly that many keys).
    """
    n_q = model.config.num_attention_heads
    past = None
    pos = 0
    seq = input_ids.shape[1]
    stops = sorted(s for s in (q_stops or []) if 0 < s <= seq)
    q_snaps: dict[int, list] = {}
    t0 = time.perf_counter()
    while pos < seq:
        past_len = 0 if past is None else past.get_seq_length()
        chunk = choose_chunk(past_len, seq - pos, n_q, budget_bytes)
        for stop in stops:
            if stop > pos:
                chunk = min(chunk, stop - pos)
                break
        hit = pos + chunk
        capture_flag["on"] = hit in stops or hit == seq
        ids = input_ids[:, pos : pos + chunk]
        out = model.model(input_ids=ids, past_key_values=past, use_cache=True)
        past = out.past_key_values
        pos += chunk
        if captured is not None and pos in stops:
            q_snaps[pos] = [t.clone() for t in captured]
        dt = time.perf_counter() - t0
        print(
            f"prefill {pos}/{seq} chunk={chunk} elapsed={dt:.1f}s tok/s={pos / dt:.1f}",
            flush=True,
        )
    capture_flag["on"] = False
    return past, q_snaps


def layer0_keys(past) -> torch.Tensor:
    return past.layers[0].keys


def verify(model, input_ids: torch.Tensor, budget_bytes: int) -> None:
    n = min(128, input_ids.shape[1])
    ids = input_ids[:, :n]
    print(f"verify full vs chunked prefill on {n} tokens", flush=True)
    full = model.model(input_ids=ids, use_cache=True).past_key_values
    chunked, _ = prefill(model, ids, budget_bytes, {"on": False})
    a = layer0_keys(full).detach()
    b = layer0_keys(chunked).detach()
    if a.shape != b.shape or a.dtype != b.dtype:
        raise SystemExit(f"layer0 KV shape/dtype mismatch {a.shape} {a.dtype} vs {b.shape} {b.dtype}")
    same = torch.equal(a, b)
    # Value cache at layer 0 is also a function of the token embedding only.
    va = full.layers[0].values.detach()
    vb = chunked.layers[0].values.detach()
    same_v = torch.equal(va, vb)
    print(f"layer0 K dtype={a.dtype} shape={tuple(a.shape)} bitwise_equal={same}", flush=True)
    print(f"layer0 V bitwise_equal={same_v}", flush=True)
    if not same or not same_v:
        raise SystemExit("layer-0 cache diverged; chunked prefill is not faithful")
    # A later layer should at least be close. Bit-exactness is not required once
    # attention reductions differ in association, but a broken mask will not.
    ha = full.layers[min(3, len(full.layers) - 1)].keys.float()
    hb = chunked.layers[min(3, len(chunked.layers) - 1)].keys.float()
    denom = torch.maximum(ha.abs().mean(), hb.abs().mean()).clamp_min(1e-8)
    rel = (ha - hb).abs().mean() / denom
    print(f"layer3 K mean_abs_rel={float(rel):.6e}", flush=True)
    if float(rel) > 1e-2:
        raise SystemExit("deeper-layer cache diverged beyond bf16 noise; check the mask")


def write_dump(path: Path, k: np.ndarray, v: np.ndarray, q: np.ndarray, meta: dict) -> None:
    # k, v: [layers, kv_heads, seq, dim] uint16
    # q: [layers, q_heads, dim] uint16
    header = np.zeros(8, dtype="<u4")
    magic = np.frombuffer(MAGIC, dtype="<u4")[0]
    header[0] = magic
    header[1] = VERSION
    header[2] = k.shape[0]
    header[3] = k.shape[1]
    header[4] = q.shape[1]
    header[5] = k.shape[3]
    header[6] = k.shape[2]
    header[7] = 0
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as f:
        f.write(header.tobytes())
        f.write(np.ascontiguousarray(k, dtype="<u2").tobytes())
        f.write(np.ascontiguousarray(v, dtype="<u2").tobytes())
        f.write(np.ascontiguousarray(q, dtype="<u2").tobytes())
    path.with_suffix(".meta.json").write_text(json.dumps(meta, indent=2) + "\n")
    print(f"wrote {path} ({path.stat().st_size} bytes)", flush=True)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="Qwen/Qwen2.5-0.5B")
    p.add_argument("--revision", default="060db6499f32faf8b98477b0a26969ef7d8b9987")
    p.add_argument("--text", type=Path, required=True)
    p.add_argument("--seq-len", type=int, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--budget-mb", type=int, default=400)
    p.add_argument("--verify-only", action="store_true")
    p.add_argument("--skip-verify", action="store_true")
    args = p.parse_args()

    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)
    text = args.text.read_text(encoding="utf-8", errors="replace")
    tokenizer = AutoTokenizer.from_pretrained(args.model, revision=args.revision)
    ids = tokenizer(text, return_tensors="pt", add_special_tokens=True).input_ids
    print(f"document tokens available: {ids.shape[1]}", flush=True)
    if ids.shape[1] < args.seq_len:
        raise SystemExit(f"need {args.seq_len} tokens, document has {ids.shape[1]}")
    input_ids = ids[:, : args.seq_len].contiguous()

    print("loading model", flush=True)
    t0 = time.perf_counter()
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        revision=args.revision,
        dtype=torch.bfloat16,
    )
    model.eval()
    print(f"loaded in {time.perf_counter() - t0:.1f}s attn={model.config._attn_implementation}", flush=True)
    print(
        f"layers={model.config.num_hidden_layers} q_heads={model.config.num_attention_heads} "
        f"kv_heads={model.config.num_key_value_heads} hidden={model.config.hidden_size} "
        f"sliding={getattr(model.config, 'use_sliding_window', None)} "
        f"layer_types={getattr(model.config, 'layer_types', None)}",
        flush=True,
    )
    budget = args.budget_mb * 1024 * 1024
    if not args.skip_verify:
        with torch.no_grad():
            verify(model, input_ids, budget)
    if args.verify_only:
        return

    captured: list = [None] * model.config.num_hidden_layers
    capture_flag = {"on": False}
    install_q_hooks(model, captured, capture_flag)
    stops = sorted({s for s in (8192, 16384, args.seq_len) if s <= args.seq_len})
    print(f"prefill seq_len={args.seq_len} q_stops={stops}", flush=True)
    with torch.no_grad():
        past, q_snaps = prefill(
            model, input_ids, budget, capture_flag, captured=captured, q_stops=stops
        )

    k0 = past.layers[0].keys
    print(f"cache dtype={k0.dtype} shape={tuple(k0.shape)}", flush=True)
    if k0.dtype != torch.bfloat16:
        print("WARNING: cache dtype is not bfloat16; casting with round-to-nearest-even", flush=True)

    layers = model.config.num_hidden_layers
    kv_heads = model.config.num_key_value_heads
    q_heads = model.config.num_attention_heads
    seq = args.seq_len
    dim = model.config.hidden_size // q_heads
    k = np.empty((layers, kv_heads, seq, dim), dtype=np.uint16)
    v = np.empty((layers, kv_heads, seq, dim), dtype=np.uint16)
    for i, layer in enumerate(past.layers):
        ki = to_u16(layer.keys[0])
        vi = to_u16(layer.values[0])
        if ki.shape != (kv_heads, seq, dim):
            raise SystemExit(f"unexpected K shape at layer {i}: {ki.shape}")
        k[i] = ki
        v[i] = vi
        print(f"copied layer {i}", flush=True)
    if any(t is None for t in captured):
        raise SystemExit("failed to capture last-position queries")
    q = np.stack([to_u16(t) for t in captured], axis=0)
    qsnap_path = args.out.with_suffix(".qsnap")
    snap_positions = sorted(q_snaps)
    if snap_positions[-1] != seq or any(any(t is None for t in q_snaps[p]) for p in snap_positions):
        raise SystemExit(f"incomplete query snapshots: {snap_positions}")
    qsnap = np.stack(
        [np.stack([to_u16(t) for t in q_snaps[p]], axis=0) for p in snap_positions],
        axis=0,
    )
    meta = {
        "model": args.model,
        "revision": args.revision,
        "torch": torch.__version__,
        "seq_len": seq,
        "n_layers": layers,
        "n_kv_heads": kv_heads,
        "n_q_heads": q_heads,
        "head_dim": dim,
        "cache_dtype_before_cast": str(k0.dtype),
        "stored_dtype": "bfloat16",
        "text_file": str(args.text),
        "text_source": "Project Gutenberg ebook 1342, Pride and Prejudice",
        "text_url": "https://www.gutenberg.org/cache/epub/1342/pg1342.txt",
        "n_tokens_in_document": int(ids.shape[1]),
        "layout": {
            "magic": "KVD1",
            "header_u32": [
                "magic",
                "version",
                "n_layers",
                "n_kv_heads",
                "n_q_heads",
                "head_dim",
                "seq_len",
                "reserved",
            ],
            "K": "[n_layers, n_kv_heads, seq_len, head_dim] uint16 little-endian bf16 bits",
            "V": "[n_layers, n_kv_heads, seq_len, head_dim] uint16 little-endian bf16 bits",
            "Q_last": "[n_layers, n_q_heads, head_dim] uint16, post-RoPE query at the last position",
        "Q_snapshots": "[n_snap, n_layers, n_q_heads, head_dim] uint16 in the .qsnap file; positions are token counts the query attends over",
        },
        "q_snapshot_positions": snap_positions,
        "q_snapshot_file": qsnap_path.name,
    }
    write_dump(args.out, k, v, q, meta)
    qsnap_header = np.zeros(4 + len(snap_positions), dtype="<u4")
    qsnap_header[0] = np.frombuffer(b"QSN1", dtype="<u4")[0]
    qsnap_header[1] = len(snap_positions)
    qsnap_header[2] = layers
    qsnap_header[3] = q_heads
    # dim is already in the kvdump; repeat it after positions via the meta. Store positions next.
    with qsnap_path.open("wb") as f:
        f.write(qsnap_header[:4].tobytes())
        f.write(np.asarray(snap_positions, dtype="<u4").tobytes())
        f.write(np.asarray([dim], dtype="<u4").tobytes())
        f.write(np.ascontiguousarray(qsnap, dtype="<u2").tobytes())
    print(f"wrote {qsnap_path} positions={snap_positions}", flush=True)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.exit(130)
