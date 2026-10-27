// SPDX-License-Identifier: MIT
#pragma once

#include <cstdint>
#include <cstring>

namespace kvexp {

// BF16 is the high 16 bits of an IEEE-754 binary32 value.
// Layout: sign 1 | exponent 8 | mantissa 7.

inline constexpr int kEncodeBlock = 128;
inline constexpr int kMaxHeadDim = 256;

inline float bf16_to_fp32(uint16_t bits) {
    uint32_t u = uint32_t(bits) << 16;
    float f;
    std::memcpy(&f, &u, sizeof(f));
    return f;
}

inline uint8_t bf16_exponent(uint16_t bits) {
    return uint8_t((bits >> 7) & 0xFF);
}

inline uint8_t bf16_sm(uint16_t bits) {
    // Sign in bit 7, mantissa in bits 6:0. The exponent is not stored.
    return uint8_t(((bits >> 8) & 0x80) | (bits & 0x7F));
}

inline uint16_t bf16_from_parts(uint8_t exponent, uint8_t sm) {
    return uint16_t((uint16_t(sm & 0x80) << 8) | (uint16_t(exponent) << 7) | uint16_t(sm & 0x7F));
}

inline uint16_t bf16_from_delta(uint8_t base, uint8_t delta, uint8_t sm) {
    return bf16_from_parts(uint8_t(uint16_t(base) + uint16_t(delta)), sm);
}

inline int exp_payload_bytes(int n_tokens, int dim, int bits) {
    return (n_tokens * dim * bits + 7) / 8;
}

inline int coded_block_bytes(int n_tokens, int dim, int bits, int n_escape) {
    return 8 + dim + n_tokens * dim + exp_payload_bytes(n_tokens, dim, bits) + 2 * n_escape;
}

inline int raw_block_bytes(int n_tokens, int dim) {
    return 8 + 2 * n_tokens * dim;
}

} // namespace kvexp
