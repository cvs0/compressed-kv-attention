// SPDX-License-Identifier: MIT
#pragma once

#include <cstddef>
#include <cstdint>

namespace kvexp::detail {

void range_scalar(const uint16_t *q, const uint16_t *k_raw, const uint16_t *v_raw,
                  const uint8_t *k_xkv, size_t k_bytes, const uint8_t *v_xkv, size_t v_bytes,
                  int n_q, int dim, int seq, int t0, int t1, float *m, float *l, float *acc);

void range_avx2(const uint16_t *q, const uint16_t *k_raw, const uint16_t *v_raw,
                const uint8_t *k_xkv, size_t k_bytes, const uint8_t *v_xkv, size_t v_bytes, int n_q,
                int dim, int seq, int t0, int t1, float *m, float *l, float *acc);

void range_avx512(const uint16_t *q, const uint16_t *k_raw, const uint16_t *v_raw,
                  const uint8_t *k_xkv, size_t k_bytes, const uint8_t *v_xkv, size_t v_bytes,
                  int n_q, int dim, int seq, int t0, int t1, float *m, float *l, float *acc);

} // namespace kvexp::detail
