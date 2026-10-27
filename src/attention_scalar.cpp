// SPDX-License-Identifier: MIT

#include "detail.hpp"
#include "kvexp/scan.hpp"
#include "softmax.hpp"

#include <stdexcept>
#include <vector>

namespace kvexp::detail {
namespace {

float hsum8(const float lane[8]) {
    const float a = (lane[0] + lane[4]) + (lane[1] + lane[5]);
    const float b = (lane[2] + lane[6]) + (lane[3] + lane[7]);
    return a + b;
}

void apply_token(const float *qfp, const uint16_t *k, const uint16_t *v, int n_q, int dim,
                 float *score, float *alpha, float *e, float *m, float *l, float *acc) {
    for (int g = 0; g < n_q; ++g) {
        float lane[8] = {};
        const float *qg = qfp + size_t(g) * size_t(dim);
        for (int d = 0; d < dim; d += 8) {
            for (int j = 0; j < 8; ++j) {
                lane[j] = std::fmaf(qg[d + j], bf16_to_fp32(k[d + j]), lane[j]);
            }
        }
        score[g] = hsum8(lane);
    }
    for (int g = 0; g < n_q; ++g) {
        softmax_coeff(m[g], l[g], score[g], alpha[g], e[g]);
    }
    for (int g = 0; g < n_q; ++g) {
        float *ag = acc + size_t(g) * size_t(dim);
        const float eg = e[g];
        const float al = alpha[g];
        for (int d = 0; d < dim; ++d) {
            ag[d] = std::fmaf(eg, bf16_to_fp32(v[d]), ag[d] * al);
        }
    }
}

} // namespace

void range_scalar(const uint16_t *q, const uint16_t *k_raw, const uint16_t *v_raw,
                  const uint8_t *k_xkv, size_t k_bytes, const uint8_t *v_xkv, size_t v_bytes,
                  int n_q, int dim, int seq, int t0, int t1, float *m, float *l, float *acc) {
    (void)seq;
    if (dim % 8 != 0 || dim <= 0 || dim > kMaxHeadDim || n_q <= 0) {
        throw std::invalid_argument("scalar attention: bad shape");
    }
    if (t0 >= t1) {
        return;
    }
    std::vector<float> qfp(size_t(n_q) * size_t(dim));
    for (size_t i = 0; i < qfp.size(); ++i) {
        qfp[i] = bf16_to_fp32(q[i]);
    }
    std::vector<float> score(n_q);
    std::vector<float> alpha(n_q);
    std::vector<float> e(n_q);
    std::vector<uint16_t> kt(dim);
    std::vector<uint16_t> vt(dim);

    if (k_raw != nullptr) {
        for (int t = t0; t < t1; ++t) {
            apply_token(qfp.data(), k_raw + size_t(t) * dim, v_raw + size_t(t) * dim, n_q, dim,
                        score.data(), alpha.data(), e.data(), m, l, acc);
        }
        return;
    }

    XkvReader kc(k_xkv, k_bytes, dim, t0);
    XkvReader vc(v_xkv, v_bytes, dim, t0);
    for (int t = t0; t < t1; ++t) {
        (void)t;
        for (int d = 0; d < dim; ++d) {
            kt[size_t(d)] = kc.read_next();
            vt[size_t(d)] = vc.read_next();
        }
        apply_token(qfp.data(), kt.data(), vt.data(), n_q, dim, score.data(), alpha.data(),
                    e.data(), m, l, acc);
    }
}

} // namespace kvexp::detail
