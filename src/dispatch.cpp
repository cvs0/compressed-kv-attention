// SPDX-License-Identifier: MIT

#include "detail.hpp"
#include "kvexp/attention.hpp"
#include "softmax.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <omp.h>
#include <stdexcept>
#include <vector>

namespace kvexp {
namespace {

void token_bounds(int seq, int tile, int tid, int nthreads, int &a, int &b) {
    const int ntiles = (seq + tile - 1) / tile;
    const int b0 = int((long long)tid * ntiles / nthreads);
    const int b1 = int((long long)(tid + 1) * ntiles / nthreads);
    a = std::min(seq, b0 * tile);
    b = std::min(seq, b1 * tile);
}

} // namespace

void attention_decode(Backend backend, const uint16_t *q, const uint16_t *k_raw,
                      const uint16_t *v_raw, const uint8_t *k_xkv, size_t k_bytes,
                      const uint8_t *v_xkv, size_t v_bytes, float *out, int n_q, int dim, int seq,
                      int n_threads, int split_tokens) {
    if (q == nullptr || out == nullptr || n_q <= 0 || dim <= 0 || seq <= 0 || split_tokens <= 0) {
        throw std::invalid_argument("attention_decode: bad argument");
    }
    const bool raw = k_raw != nullptr;
    const bool xkv = k_xkv != nullptr;
    if (raw == xkv || (v_raw != nullptr) != raw || (v_xkv != nullptr) != xkv) {
        throw std::invalid_argument("attention_decode: pass raw K and V or xkv K and V, not both");
    }
    const CpuFeatures cpu = detect_cpu();
    if (backend == Backend::Avx2 && !cpu.avx2) {
        throw std::runtime_error("AVX2 requested but not available");
    }
    if (backend == Backend::Avx512Bf16 && !cpu.avx512_bf16) {
        throw std::runtime_error("AVX-512 BF16 requested but not available");
    }

    const int ntiles = (seq + split_tokens - 1) / split_tokens;
    int threads = std::max(1, n_threads);
    threads = std::min(threads, ntiles);

    auto run = [&](int t0, int t1, float *m, float *l, float *acc) {
        switch (backend) {
        case Backend::Scalar:
            detail::range_scalar(q, k_raw, v_raw, k_xkv, k_bytes, v_xkv, v_bytes, n_q, dim, seq, t0,
                                 t1, m, l, acc);
            break;
        case Backend::Avx2:
            detail::range_avx2(q, k_raw, v_raw, k_xkv, k_bytes, v_xkv, v_bytes, n_q, dim, seq, t0,
                               t1, m, l, acc);
            break;
        case Backend::Avx512Bf16:
            detail::range_avx512(q, k_raw, v_raw, k_xkv, k_bytes, v_xkv, v_bytes, n_q, dim, seq, t0,
                                 t1, m, l, acc);
            break;
        }
    };

    const float ninf = -std::numeric_limits<float>::infinity();
    std::vector<float> m(size_t(threads) * size_t(n_q), ninf);
    std::vector<float> l(size_t(threads) * size_t(n_q), 0.f);
    std::vector<float> acc(size_t(threads) * size_t(n_q) * size_t(dim), 0.f);

    if (threads == 1) {
        run(0, seq, m.data(), l.data(), acc.data());
        detail::finalize_state(l.data(), acc.data(), out, n_q, dim);
        return;
    }

    omp_set_dynamic(0);
    int got = 0;
#pragma omp parallel num_threads(threads)
    {
#pragma omp single
        got = omp_get_num_threads();
        const int tid = omp_get_thread_num();
        int a = 0, b = 0;
        token_bounds(seq, split_tokens, tid, threads, a, b);
        float *ms = m.data() + size_t(tid) * size_t(n_q);
        float *ls = l.data() + size_t(tid) * size_t(n_q);
        float *as = acc.data() + size_t(tid) * size_t(n_q) * size_t(dim);
        if (a < b) {
            run(a, b, ms, ls, as);
        }
    }
    if (got != threads) {
        throw std::runtime_error("OpenMP did not start the requested thread count");
    }
    for (int t = 1; t < threads; ++t) {
        detail::merge_state(m.data(), l.data(), acc.data(), m.data() + size_t(t) * n_q,
                            l.data() + size_t(t) * n_q, acc.data() + size_t(t) * n_q * dim, n_q,
                            dim);
    }
    detail::finalize_state(l.data(), acc.data(), out, n_q, dim);
}

} // namespace kvexp
