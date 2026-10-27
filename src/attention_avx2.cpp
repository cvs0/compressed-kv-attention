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

float hsum256(__m256 v) {
    __m128 lo = _mm256_castps256_ps128(v);
    __m128 hi = _mm256_extractf128_ps(v, 1);
    __m128 s = _mm_add_ps(lo, hi);
    s = _mm_hadd_ps(s, s);
    s = _mm_hadd_ps(s, s);
    return _mm_cvtss_f32(s);
}

__m256 load8_raw(const uint16_t *p) {
    __m128i h = _mm_loadu_si128(reinterpret_cast<const __m128i *>(p));
    __m256i w = _mm256_cvtepu16_epi32(h);
    w = _mm256_slli_epi32(w, 16);
    return _mm256_castsi256_ps(w);
}

__m256 decode8_exp4(const uint8_t *sm, const uint8_t *exp, const uint8_t *base,
                    const uint16_t *&esc) {
    __m128i sm_i = _mm_loadl_epi64(reinterpret_cast<const __m128i *>(sm));
    uint32_t eb = 0;
    std::memcpy(&eb, exp, 4);
    __m128i nib = _mm_cvtsi32_si128(static_cast<int>(eb));
    const __m128i mlo = _mm_set1_epi8(0x0F);
    const __m128i mhi = _mm_set1_epi8(static_cast<char>(0xF0));
    __m128i lo = _mm_and_si128(nib, mlo);
    __m128i hi = _mm_and_si128(_mm_srli_epi16(_mm_and_si128(nib, mhi), 4), mlo);
    __m128i deltas = _mm_unpacklo_epi8(lo, hi);
    __m128i base_i = _mm_loadl_epi64(reinterpret_cast<const __m128i *>(base));
    __m128i ex = _mm_add_epi8(base_i, deltas);

    __m256i sm32 = _mm256_cvtepu8_epi32(sm_i);
    __m256i ex32 = _mm256_cvtepu8_epi32(ex);
    __m256i sign = _mm256_and_si256(sm32, _mm256_set1_epi32(0x80));
    __m256i mant = _mm256_and_si256(sm32, _mm256_set1_epi32(0x7F));
    __m256i bits =
        _mm256_or_si256(_mm256_slli_epi32(sign, 24),
                        _mm256_or_si256(_mm256_slli_epi32(ex32, 23), _mm256_slli_epi32(mant, 16)));
    __m256 f = _mm256_castsi256_ps(bits);

    __m128i is_esc = _mm_cmpeq_epi8(deltas, _mm_set1_epi8(15));
    if (!_mm_testz_si128(is_esc, is_esc)) {
        alignas(32) float tmp[8];
        alignas(16) uint8_t del[8];
        _mm256_store_ps(tmp, f);
        _mm_storel_epi64(reinterpret_cast<__m128i *>(del), deltas);
        for (int i = 0; i < 8; ++i) {
            if (del[i] == 15) {
                tmp[i] = bf16_to_fp32(*esc++);
            }
        }
        f = _mm256_load_ps(tmp);
    }
    return f;
}

__m256 decode8_exp2(const BlockView &blk, int local, int d, int dim, const uint16_t *&esc) {
    alignas(32) float tmp[8];
    for (int i = 0; i < 8; ++i) {
        const int index = local * dim + d + i;
        const uint8_t sm = blk.sm[index];
        const int code = residual_code(blk.exp, index, 2);
        uint16_t bf;
        if (code == 3) {
            bf = *esc++;
        } else {
            bf = bf16_from_delta(blk.base[d + i], uint8_t(code), sm);
        }
        tmp[i] = bf16_to_fp32(bf);
    }
    return _mm256_load_ps(tmp);
}

void load_token(XkvPos &pos, __m256 *regs) {
    const int dim = pos.dim;
    if (pos.blk.mode == 0) {
        const uint16_t *row = pos.blk.raw + size_t(pos.local) * size_t(dim);
        for (int i = 0, d = 0; d < dim; d += 8, ++i) {
            regs[i] = load8_raw(row + d);
        }
    } else if (pos.blk.mode == 4) {
        const uint8_t *sm = pos.blk.sm + size_t(pos.local) * size_t(dim);
        const uint8_t *exp = pos.blk.exp + size_t(pos.local) * size_t(dim) / 2u;
        for (int i = 0, d = 0; d < dim; d += 8, ++i) {
            regs[i] = decode8_exp4(sm + d, exp + d / 2, pos.blk.base + d, pos.esc);
        }
    } else if (pos.blk.mode == 2) {
        for (int i = 0, d = 0; d < dim; d += 8, ++i) {
            regs[i] = decode8_exp2(pos.blk, pos.local, d, dim, pos.esc);
        }
    } else {
        throw std::runtime_error("avx2: bad xkv mode");
    }
    pos.advance_token();
}

void apply_regs(const float *qfp, const __m256 *kreg, const __m256 *vreg, int nvec, int n_q,
                int dim, float *score, float *alpha, float *e, float *m, float *l, float *acc) {
    for (int g = 0; g < n_q; ++g) {
        __m256 a = _mm256_setzero_ps();
        const float *qg = qfp + size_t(g) * size_t(dim);
        for (int i = 0; i < nvec; ++i) {
            __m256 qv = _mm256_loadu_ps(qg + i * 8);
            a = _mm256_fmadd_ps(qv, kreg[i], a);
        }
        score[g] = hsum256(a);
    }
    for (int g = 0; g < n_q; ++g) {
        softmax_coeff(m[g], l[g], score[g], alpha[g], e[g]);
    }
    for (int i = 0; i < nvec; ++i) {
        const __m256 v = vreg[i];
        for (int g = 0; g < n_q; ++g) {
            float *ag = acc + (size_t(g) * size_t(dim) + size_t(i) * 8u);
            __m256 a = _mm256_loadu_ps(ag);
            __m256 scaled = _mm256_mul_ps(a, _mm256_set1_ps(alpha[g]));
            a = _mm256_fmadd_ps(_mm256_set1_ps(e[g]), v, scaled);
            _mm256_storeu_ps(ag, a);
        }
    }
}

} // namespace

void range_avx2(const uint16_t *q, const uint16_t *k_raw, const uint16_t *v_raw,
                const uint8_t *k_xkv, size_t k_bytes, const uint8_t *v_xkv, size_t v_bytes, int n_q,
                int dim, int seq, int t0, int t1, float *m, float *l, float *acc) {
    (void)seq;
    if (dim % 8 != 0 || dim <= 0 || dim > kMaxHeadDim || n_q <= 0) {
        throw std::invalid_argument("avx2 attention: bad shape");
    }
    if (t0 >= t1) {
        return;
    }
    const int nvec = dim / 8;
    std::vector<float> qfp(size_t(n_q) * size_t(dim));
    for (size_t i = 0; i < qfp.size(); ++i) {
        qfp[i] = bf16_to_fp32(q[i]);
    }
    std::vector<float> score(n_q);
    std::vector<float> alpha(n_q);
    std::vector<float> e(n_q);
    __m256 kreg[kMaxHeadDim / 8];
    __m256 vreg[kMaxHeadDim / 8];

    if (k_raw != nullptr) {
        for (int t = t0; t < t1; ++t) {
            const uint16_t *kr = k_raw + size_t(t) * size_t(dim);
            const uint16_t *vr = v_raw + size_t(t) * size_t(dim);
            for (int i = 0; i < nvec; ++i) {
                kreg[i] = load8_raw(kr + i * 8);
                vreg[i] = load8_raw(vr + i * 8);
            }
            apply_regs(qfp.data(), kreg, vreg, nvec, n_q, dim, score.data(), alpha.data(), e.data(),
                       m, l, acc);
        }
        return;
    }

    XkvPos kp = open_xkv(k_xkv, k_bytes, dim, t0);
    XkvPos vp = open_xkv(v_xkv, v_bytes, dim, t0);
    for (int t = t0; t < t1; ++t) {
        (void)t;
        if (kp.done || vp.done) {
            throw std::runtime_error("avx2 xkv ran out of tokens");
        }
        load_token(kp, kreg);
        load_token(vp, vreg);
        apply_regs(qfp.data(), kreg, vreg, nvec, n_q, dim, score.data(), alpha.data(), e.data(), m,
                   l, acc);
    }
}

} // namespace kvexp::detail
