// SPDX-License-Identifier: MIT

#include "detail.hpp"
#include "kvexp/scan.hpp"
#include "softmax.hpp"

#include <cstring>
#include <immintrin.h>
#include <stdexcept>
#include <vector>

namespace kvexp::detail {
namespace {

__m512bh as_bh(__m512i v) {
    __m512bh out;
    std::memcpy(&out, &v, sizeof(out));
    return out;
}

float hsum512(__m512 v) {
    // Pairwise tree over the 16 lanes. Both the raw and the coded kernel call this.
    alignas(64) float t[16];
    _mm512_store_ps(t, v);
    const float a0 = (t[0] + t[1]) + (t[2] + t[3]);
    const float a1 = (t[4] + t[5]) + (t[6] + t[7]);
    const float a2 = (t[8] + t[9]) + (t[10] + t[11]);
    const float a3 = (t[12] + t[13]) + (t[14] + t[15]);
    return (a0 + a1) + (a2 + a3);
}

__m512 bf16_16_to_ps(__m256i h) {
    __m512i w = _mm512_cvtepu16_epi32(h);
    w = _mm512_slli_epi32(w, 16);
    return _mm512_castsi512_ps(w);
}

void bf16_32_to_ps(__m512i bf, __m512 &lo, __m512 &hi) {
    __m256i l = _mm512_castsi512_si256(bf);
    __m256i h = _mm512_extracti64x4_epi64(bf, 1);
    lo = bf16_16_to_ps(l);
    hi = bf16_16_to_ps(h);
}

__m512i decode32_exp4(const uint8_t *sm, const uint8_t *exp, const uint8_t *base,
                      const uint16_t *&esc) {
    __m256i sm256 = _mm256_loadu_si256(reinterpret_cast<const __m256i *>(sm));
    __m128i exp16 = _mm_loadu_si128(reinterpret_cast<const __m128i *>(exp));
    __m256i base256 = _mm256_loadu_si256(reinterpret_cast<const __m256i *>(base));
    const __m128i mlo = _mm_set1_epi8(0x0F);
    const __m128i mhi = _mm_set1_epi8(static_cast<char>(0xF0));
    __m128i lo = _mm_and_si128(exp16, mlo);
    __m128i hi = _mm_and_si128(_mm_srli_epi16(_mm_and_si128(exp16, mhi), 4), mlo);
    __m128i d_lo = _mm_unpacklo_epi8(lo, hi);
    __m128i d_hi = _mm_unpackhi_epi8(lo, hi);
    __m256i deltas = _mm256_set_m128i(d_hi, d_lo);
    __m256i ex = _mm256_add_epi8(base256, deltas);

    __m512i sm16 = _mm512_cvtepu8_epi16(sm256);
    __m512i ex16 = _mm512_cvtepu8_epi16(ex);
    __m512i sign = _mm512_and_si512(sm16, _mm512_set1_epi16(0x80));
    __m512i mant = _mm512_and_si512(sm16, _mm512_set1_epi16(0x7F));
    __m512i bf = _mm512_or_si512(_mm512_slli_epi16(sign, 8),
                                 _mm512_or_si512(_mm512_slli_epi16(ex16, 7), mant));

    __m256i is_esc = _mm256_cmpeq_epi8(deltas, _mm256_set1_epi8(15));
    if (!_mm256_testz_si256(is_esc, is_esc)) {
        alignas(64) uint16_t tmp[32];
        alignas(32) uint8_t del[32];
        _mm512_store_si512(reinterpret_cast<__m512i *>(tmp), bf);
        _mm256_store_si256(reinterpret_cast<__m256i *>(del), deltas);
        for (int i = 0; i < 32; ++i) {
            if (del[i] == 15) {
                tmp[i] = *esc++;
            }
        }
        bf = _mm512_load_si512(reinterpret_cast<const __m512i *>(tmp));
    }
    return bf;
}

__m512i decode32_exp2(const BlockView &blk, int local, int d, int dim, const uint16_t *&esc) {
    alignas(64) uint16_t tmp[32];
    for (int i = 0; i < 32; ++i) {
        const int index = local * dim + d + i;
        const uint8_t sm = blk.sm[index];
        const int code = residual_code(blk.exp, index, 2);
        if (code == 3) {
            tmp[i] = *esc++;
        } else {
            tmp[i] = bf16_from_delta(blk.base[d + i], uint8_t(code), sm);
        }
    }
    return _mm512_load_si512(reinterpret_cast<const __m512i *>(tmp));
}

__m512i load32(XkvPos &pos, int d) {
    const int dim = pos.dim;
    if (pos.blk.mode == 0) {
        const uint16_t *row = pos.blk.raw + size_t(pos.local) * size_t(dim) + size_t(d);
        return _mm512_loadu_si512(row);
    }
    if (pos.blk.mode == 4) {
        const uint8_t *sm = pos.blk.sm + size_t(pos.local) * size_t(dim) + size_t(d);
        const uint8_t *exp = pos.blk.exp + (size_t(pos.local) * size_t(dim) + size_t(d)) / 2u;
        return decode32_exp4(sm, exp, pos.blk.base + d, pos.esc);
    }
    if (pos.blk.mode == 2) {
        return decode32_exp2(pos.blk, pos.local, d, dim, pos.esc);
    }
    throw std::runtime_error("avx512: bad xkv mode");
}

void apply512(const uint16_t *q, const __m512i *kbf, const __m512 *vps, int nk, int nv, int n_q,
              int dim, float *score, float *alpha, float *e, float *m, float *l, float *acc) {
    for (int g = 0; g < n_q; ++g) {
        __m512 a = _mm512_setzero_ps();
        const uint16_t *qg = q + size_t(g) * size_t(dim);
        for (int i = 0; i < nk; ++i) {
            __m512i qb = _mm512_loadu_si512(qg + i * 32);
            a = _mm512_dpbf16_ps(a, as_bh(kbf[i]), as_bh(qb));
        }
        score[g] = hsum512(a);
    }
    for (int g = 0; g < n_q; ++g) {
        softmax_coeff(m[g], l[g], score[g], alpha[g], e[g]);
    }
    for (int i = 0; i < nv; ++i) {
        const __m512 v = vps[i];
        for (int g = 0; g < n_q; ++g) {
            float *ag = acc + (size_t(g) * size_t(dim) + size_t(i) * 16u);
            __m512 a = _mm512_loadu_ps(ag);
            __m512 scaled = _mm512_mul_ps(a, _mm512_set1_ps(alpha[g]));
            a = _mm512_fmadd_ps(_mm512_set1_ps(e[g]), v, scaled);
            _mm512_storeu_ps(ag, a);
        }
    }
}

void load_pair(XkvPos &kp, XkvPos &vp, int dim, __m512i *kbf, __m512 *vps) {
    const int nk = dim / 32;
    for (int i = 0; i < nk; ++i) {
        kbf[i] = load32(kp, i * 32);
        __m512i vb = load32(vp, i * 32);
        bf16_32_to_ps(vb, vps[2 * i], vps[2 * i + 1]);
    }
    kp.advance_token();
    vp.advance_token();
}

} // namespace

void range_avx512(const uint16_t *q, const uint16_t *k_raw, const uint16_t *v_raw,
                  const uint8_t *k_xkv, size_t k_bytes, const uint8_t *v_xkv, size_t v_bytes,
                  int n_q, int dim, int seq, int t0, int t1, float *m, float *l, float *acc) {
    (void)seq;
    if (dim % 32 != 0 || dim <= 0 || dim > kMaxHeadDim || n_q <= 0) {
        throw std::invalid_argument("avx512 attention: head_dim must be a multiple of 32");
    }
    if (t0 >= t1) {
        return;
    }
    const int nk = dim / 32;
    const int nv = dim / 16;
    std::vector<float> score(n_q);
    std::vector<float> alpha(n_q);
    std::vector<float> e(n_q);
    __m512i kbf[kMaxHeadDim / 32];
    __m512 vps[kMaxHeadDim / 16];

    if (k_raw != nullptr) {
        for (int t = t0; t < t1; ++t) {
            const uint16_t *kr = k_raw + size_t(t) * size_t(dim);
            const uint16_t *vr = v_raw + size_t(t) * size_t(dim);
            for (int i = 0; i < nk; ++i) {
                kbf[i] = _mm512_loadu_si512(kr + i * 32);
                __m512i vb = _mm512_loadu_si512(vr + i * 32);
                bf16_32_to_ps(vb, vps[2 * i], vps[2 * i + 1]);
            }
            apply512(q, kbf, vps, nk, nv, n_q, dim, score.data(), alpha.data(), e.data(), m, l,
                     acc);
        }
        return;
    }

    XkvPos kp = open_xkv(k_xkv, k_bytes, dim, t0);
    XkvPos vp = open_xkv(v_xkv, v_bytes, dim, t0);
    for (int n = t0; n < t1; ++n) {
        (void)n;
        if (kp.done || vp.done) {
            throw std::runtime_error("avx512 xkv ran out of tokens");
        }
        load_pair(kp, vp, dim, kbf, vps);
        apply512(q, kbf, vps, nk, nv, n_q, dim, score.data(), alpha.data(), e.data(), m, l, acc);
    }
}

} // namespace kvexp::detail
