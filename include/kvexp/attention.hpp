// SPDX-License-Identifier: MIT
#pragma once

#include "kvexp/bf16.hpp"

#include <cstdint>

namespace kvexp {

enum class Backend {
    Scalar,
    Avx2,
    Avx512Bf16,
};

struct CpuFeatures {
    bool avx2 = false;
    bool avx512_bf16 = false;
};

CpuFeatures detect_cpu();

// One decode-attention step over `seq` keys for `n_q` queries that share one
// KV head. Q, K, and V are BF16. Output is FP32.
//
// Exactly one of (k_raw, k_xkv) is non-null, and the same for V. Raw K and V
// are [seq, dim] token-major. XKV blobs are `encode_bf16` containers of that
// same matrix.
//
// Accumulation, scalar and AVX2 (dim multiple of 8):
//   lane[j] = 0 for j in 0..7
//   for d = 0, 8, 16, ...:
//     lane[j] = fmaf(q[d+j], k[d+j], lane[j])     // BF16 widened to FP32
//   score = ((lane0+lane4)+(lane1+lane5)) + ((lane2+lane6)+(lane3+lane7))
//   m' = max(m, score)
//   alpha = exp(m - m'); e = exp(score - m')
//   l = fmaf(l, alpha, e); m = m'
//   acc[d] = fmaf(e, v[d], acc[d] * alpha)
//   out = acc / l
// AVX-512 BF16 uses VDPBF16PS for the score (32 BF16 values per step, then a
// horizontal sum) and the same FP32 V update at width 16. It matches the
// uncompressed AVX-512 kernel, not the scalar kernel.
//
// `n_threads` splits the sequence on `split_tokens` boundaries. Those
// boundaries must be the block size used to encode XKV (the tail may be
// short). Thread-local online-softmax states are folded in thread order with
//   m = max(m1, m2)
//   e1 = exp(m1-m), e2 = exp(m2-m)
//   l = fmaf(l1, e1, l2*e2)
//   acc = fmaf(acc1, e1, acc2*e2)
// Single-thread and multi-thread results are not required to be bit-identical
// to each other. Compressed and raw results for the same backend, thread
// count, and split are bit-identical.
//
// The process MXCSR is left unchanged. On Linux x86-64 the ABI default has
// FTZ and DAZ set, and both kernels observe that state.

void attention_decode(Backend backend, const uint16_t *q, const uint16_t *k_raw,
                      const uint16_t *v_raw, const uint8_t *k_xkv, size_t k_bytes,
                      const uint8_t *v_xkv, size_t v_bytes, float *out, int n_q, int dim, int seq,
                      int n_threads, int split_tokens = kEncodeBlock);

} // namespace kvexp
