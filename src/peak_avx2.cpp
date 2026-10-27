// SPDX-License-Identifier: MIT

#include "peak.hpp"

#include <chrono>
#include <cstdint>
#include <immintrin.h>
#include <omp.h>

namespace {

double seconds_since(std::chrono::steady_clock::time_point t0) {
    return std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
}

} // namespace

double peak_avx2_fp32_gflops(int threads) {
    if (threads < 1) {
        threads = 1;
    }
    // 8 independent 8-wide chains: 128 flops per iteration per thread.
    constexpr double kFlopsPerIter = 128.0;
    auto run = [&](int64_t iters) {
        const auto t0 = std::chrono::steady_clock::now();
        omp_set_dynamic(0);
#pragma omp parallel num_threads(threads)
        {
            __m256 a0 = _mm256_set1_ps(0.10f);
            __m256 a1 = _mm256_set1_ps(0.11f);
            __m256 a2 = _mm256_set1_ps(0.12f);
            __m256 a3 = _mm256_set1_ps(0.13f);
            __m256 a4 = _mm256_set1_ps(0.14f);
            __m256 a5 = _mm256_set1_ps(0.15f);
            __m256 a6 = _mm256_set1_ps(0.16f);
            __m256 a7 = _mm256_set1_ps(0.17f);
            const __m256 b = _mm256_set1_ps(1.0000001f);
            const __m256 c = _mm256_set1_ps(0.9999999f);
            int64_t n = iters;
            asm volatile("1:\n\t"
                         "vfmadd231ps %[b], %[c], %[a0]\n\t"
                         "vfmadd231ps %[b], %[c], %[a1]\n\t"
                         "vfmadd231ps %[b], %[c], %[a2]\n\t"
                         "vfmadd231ps %[b], %[c], %[a3]\n\t"
                         "vfmadd231ps %[b], %[c], %[a4]\n\t"
                         "vfmadd231ps %[b], %[c], %[a5]\n\t"
                         "vfmadd231ps %[b], %[c], %[a6]\n\t"
                         "vfmadd231ps %[b], %[c], %[a7]\n\t"
                         "subq $1, %[n]\n\t"
                         "jnz 1b\n\t"
                         : [a0] "+v"(a0), [a1] "+v"(a1), [a2] "+v"(a2), [a3] "+v"(a3),
                           [a4] "+v"(a4), [a5] "+v"(a5), [a6] "+v"(a6), [a7] "+v"(a7), [n] "+r"(n)
                         : [b] "v"(b), [c] "v"(c)
                         : "cc");
            __m256 s = _mm256_add_ps(_mm256_add_ps(a0, a1), _mm256_add_ps(a2, a3));
            s = _mm256_add_ps(s, _mm256_add_ps(_mm256_add_ps(a4, a5), _mm256_add_ps(a6, a7)));
            alignas(32) float sink[8];
            _mm256_store_ps(sink, s);
            volatile float keep = sink[0] + sink[7];
            (void)keep;
        }
        return seconds_since(t0);
    };
    int64_t iters = 20000;
    double sec = run(iters);
    if (sec < 1e-4) {
        sec = 1e-4;
    }
    iters = static_cast<int64_t>(iters * (0.20 / sec));
    if (iters < 20000) {
        iters = 20000;
    }
    sec = run(iters);
    const double flops = double(threads) * double(iters) * kFlopsPerIter;
    return (flops / 1e9) / sec;
}
