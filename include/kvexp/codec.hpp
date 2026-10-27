// SPDX-License-Identifier: MIT
#pragma once

#include "kvexp/bf16.hpp"

#include <cstddef>
#include <cstdint>
#include <vector>

namespace kvexp {

// Lossless fixed-rate BF16 exponent coding.
//
// A tensor is [seq, dim] BF16, packed as a sequence of blocks with no outer
// header and no padding. Each block covers up to `block_tokens` consecutive
// positions (the last block may be short).
//
// Block, little-endian:
//   u8  mode          0 = raw BF16, 2 = 2-bit residuals, 4 = 4-bit residuals
//   u8  n_tokens
//   u16 n_escape      0 when mode == 0
//   u16 head_dim
//   u16 reserved      0
// Raw mode payload: n_tokens * head_dim little-endian BF16, token-major.
// Coded payload:
//   u8 base[head_dim]                 per-dimension window base
//   u8 sm[n_tokens * head_dim]        sign in bit 7, mantissa in bits 6:0
//   u8 exp_packed[...]                `bits` per element, low bits of the
//                                     first element in the low bits of a byte
//   u16 escape[n_escape]              full BF16, scan order
// Element order is token-major. For a coded element the stored code is
// (exponent - base[d]) when that residual is in 0 .. 2^bits-2. The all-ones
// code is the escape sentinel and the original BF16 is the next escape word.
// The base of a dimension is the smallest exponent that maximises how many
// values fall in [base, base + 2^bits - 2]. A block uses the mode whose
// container is smallest; ties prefer raw, then fewer escapes, then fewer bits.

struct EncodedTensor {
    std::vector<uint8_t> bytes;
    int seq_len = 0;
    int head_dim = 0;
    int block_tokens = kEncodeBlock;
};

EncodedTensor encode_bf16(const uint16_t *kv, int seq_len, int head_dim,
                          int block_tokens = kEncodeBlock);

void decode_bf16(const uint8_t *blob, size_t bytes, int seq_len, int head_dim, uint16_t *out);

} // namespace kvexp
