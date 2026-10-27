// SPDX-License-Identifier: MIT
#pragma once

#include <cmath>

namespace kvexp::detail {

inline void softmax_coeff(float &m, float &l, float score, float &alpha, float &e) {
    const float m2 = std::max(m, score);
    alpha = std::exp(m - m2);
    e = std::exp(score - m2);
    l = std::fmaf(l, alpha, e);
    m = m2;
}

inline void merge_state(float *m, float *l, float *acc, const float *mb, const float *lb,
                        const float *accb, int n_q, int dim) {
    for (int g = 0; g < n_q; ++g) {
        const float mm = std::max(m[g], mb[g]);
        const float e1 = std::exp(m[g] - mm);
        const float e2 = std::exp(mb[g] - mm);
        l[g] = std::fmaf(l[g], e1, lb[g] * e2);
        float *a = acc + size_t(g) * size_t(dim);
        const float *b = accb + size_t(g) * size_t(dim);
        for (int d = 0; d < dim; ++d) {
            a[d] = std::fmaf(a[d], e1, b[d] * e2);
        }
        m[g] = mm;
    }
}

inline void finalize_state(const float *l, const float *acc, float *out, int n_q, int dim) {
    for (int g = 0; g < n_q; ++g) {
        const float inv = 1.0f / l[g];
        const float *a = acc + size_t(g) * size_t(dim);
        float *o = out + size_t(g) * size_t(dim);
        for (int d = 0; d < dim; ++d) {
            o[d] = a[d] * inv;
        }
    }
}

} // namespace kvexp::detail
