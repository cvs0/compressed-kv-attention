// SPDX-License-Identifier: MIT

#include "peak.hpp"

#include <chrono>
#include <cstdint>
#include <cstring>
#include <immintrin.h>
#include <omp.h>
#include <vector>

namespace {

double seconds_since(std::chrono::steady_clock::time_point t0) {
    return std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
}

template <class F> double calibrate(F &&run) {
    int64_t iters = 10000;
    double sec = run(iters);
    if (sec < 1e-4) {
        sec = 1e-4;
    }
    iters = static_cast<int64_t>(iters * (0.20 / sec));
    if (iters < 10000) {
        iters = 10000;
    }
    sec = run(iters);
    return sec > 0 ? double(iters) / sec : 0.0;
}

} // namespace

double peak_avx512_fp32_gflops(int threads) {
    if (threads < 1) {
        threads = 1;
    }
    // 8 independent 16-wide chains: 256 flops per iteration per thread.
    constexpr double kFlopsPerIter = 256.0;
    auto run = [&](int64_t iters) {
        const auto t0 = std::chrono::steady_clock::now();
        omp_set_dynamic(0);
#pragma omp parallel num_threads(threads)
        {
            __m512 a0 = _mm512_set1_ps(0.10f);
            __m512 a1 = _mm512_set1_ps(0.11f);
            __m512 a2 = _mm512_set1_ps(0.12f);
            __m512 a3 = _mm512_set1_ps(0.13f);
            __m512 a4 = _mm512_set1_ps(0.14f);
            __m512 a5 = _mm512_set1_ps(0.15f);
            __m512 a6 = _mm512_set1_ps(0.16f);
            __m512 a7 = _mm512_set1_ps(0.17f);
            const __m512 b = _mm512_set1_ps(1.0000001f);
            const __m512 c = _mm512_set1_ps(0.9999999f);
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
            __m512 s = _mm512_add_ps(_mm512_add_ps(a0, a1), _mm512_add_ps(a2, a3));
            s = _mm512_add_ps(s, _mm512_add_ps(_mm512_add_ps(a4, a5), _mm512_add_ps(a6, a7)));
            alignas(64) float sink[16];
            _mm512_store_ps(sink, s);
            volatile float keep = sink[0] + sink[15];
            (void)keep;
        }
        return seconds_since(t0);
    };
    const double per_sec = calibrate(run);
    return (double(threads) * kFlopsPerIter * per_sec) / 1e9;
}

double peak_avx512_dpbf16_gflops(int threads) {
    if (threads < 1) {
        threads = 1;
    }
    // 8 independent VDPBF16PS: 64 flops each.
    constexpr double kFlopsPerIter = 512.0;
    auto run = [&](int64_t iters) {
        const auto t0 = std::chrono::steady_clock::now();
        omp_set_dynamic(0);
#pragma omp parallel num_threads(threads)
        {
            alignas(64) uint16_t ones[32];
            for (int i = 0; i < 32; ++i) {
                ones[i] = 0x3F80;
            }
            __m512i b;
            __m512i c;
            std::memcpy(&b, ones, 64);
            std::memcpy(&c, ones, 64);
            __m512 a0 = _mm512_set1_ps(0.10f);
            __m512 a1 = _mm512_set1_ps(0.11f);
            __m512 a2 = _mm512_set1_ps(0.12f);
            __m512 a3 = _mm512_set1_ps(0.13f);
            __m512 a4 = _mm512_set1_ps(0.14f);
            __m512 a5 = _mm512_set1_ps(0.15f);
            __m512 a6 = _mm512_set1_ps(0.16f);
            __m512 a7 = _mm512_set1_ps(0.17f);
            int64_t n = iters;
            asm volatile("1:\n\t"
                         "vdpbf16ps %[b], %[c], %[a0]\n\t"
                         "vdpbf16ps %[b], %[c], %[a1]\n\t"
                         "vdpbf16ps %[b], %[c], %[a2]\n\t"
                         "vdpbf16ps %[b], %[c], %[a3]\n\t"
                         "vdpbf16ps %[b], %[c], %[a4]\n\t"
                         "vdpbf16ps %[b], %[c], %[a5]\n\t"
                         "vdpbf16ps %[b], %[c], %[a6]\n\t"
                         "vdpbf16ps %[b], %[c], %[a7]\n\t"
                         "subq $1, %[n]\n\t"
                         "jnz 1b\n\t"
                         : [a0] "+v"(a0), [a1] "+v"(a1), [a2] "+v"(a2), [a3] "+v"(a3),
                           [a4] "+v"(a4), [a5] "+v"(a5), [a6] "+v"(a6), [a7] "+v"(a7), [n] "+r"(n)
                         : [b] "v"(b), [c] "v"(c)
                         : "cc");
            __m512 s = _mm512_add_ps(_mm512_add_ps(a0, a1), _mm512_add_ps(a2, a3));
            s = _mm512_add_ps(s, _mm512_add_ps(_mm512_add_ps(a4, a5), _mm512_add_ps(a6, a7)));
            alignas(64) float sink[16];
            _mm512_store_ps(sink, s);
            volatile float keep = sink[0] + sink[15];
            (void)keep;
        }
        return seconds_since(t0);
    };
    const double per_sec = calibrate(run);
    return (double(threads) * kFlopsPerIter * per_sec) / 1e9;
}

double avx512_read_gbps(const void *p, size_t bytes, int nontemporal, int threads) {
    if (threads < 1) {
        threads = 1;
    }
    const size_t n = bytes / 64;
    auto *base = static_cast<__m512i *>(const_cast<void *>(p));
    auto load_range = [&](size_t begin, size_t end) {
        __m512i s0 = _mm512_setzero_si512();
        __m512i s1 = _mm512_setzero_si512();
        __m512i s2 = _mm512_setzero_si512();
        __m512i s3 = _mm512_setzero_si512();
        size_t i = begin;
        if (nontemporal) {
            for (; i + 3 < end; i += 4) {
                s0 = _mm512_add_epi64(s0, _mm512_stream_load_si512(base + i));
                s1 = _mm512_add_epi64(s1, _mm512_stream_load_si512(base + i + 1));
                s2 = _mm512_add_epi64(s2, _mm512_stream_load_si512(base + i + 2));
                s3 = _mm512_add_epi64(s3, _mm512_stream_load_si512(base + i + 3));
            }
        } else {
            for (; i + 3 < end; i += 4) {
                s0 = _mm512_add_epi64(s0, _mm512_load_si512(base + i));
                s1 = _mm512_add_epi64(s1, _mm512_load_si512(base + i + 1));
                s2 = _mm512_add_epi64(s2, _mm512_load_si512(base + i + 2));
                s3 = _mm512_add_epi64(s3, _mm512_load_si512(base + i + 3));
            }
        }
        s0 = _mm512_add_epi64(_mm512_add_epi64(s0, s1), _mm512_add_epi64(s2, s3));
        return _mm512_reduce_add_epi64(s0);
    };
    auto once = [&]() {
        if (threads == 1) {
            return load_range(0, n);
        }
        std::vector<long long> part(size_t(threads), 0);
        omp_set_dynamic(0);
#pragma omp parallel num_threads(threads)
        {
            const int tid = omp_get_thread_num();
            const int nth = omp_get_num_threads();
            const size_t a = (n * size_t(tid)) / size_t(nth);
            const size_t b = (n * size_t(tid + 1)) / size_t(nth);
            part[size_t(tid)] = load_range(a, b);
        }
        long long s = 0;
        for (long long v : part) {
            s ^= v;
        }
        return s;
    };
    volatile long long sink = once();
    (void)sink;
    double samples[5];
    for (int r = 0; r < 5; ++r) {
        const auto t0 = std::chrono::steady_clock::now();
        sink = once();
        samples[r] = seconds_since(t0);
    }
    for (int i = 0; i < 5; ++i) {
        for (int j = i + 1; j < 5; ++j) {
            if (samples[j] < samples[i]) {
                const double tmp = samples[i];
                samples[i] = samples[j];
                samples[j] = tmp;
            }
        }
    }
    const double sec = samples[2];
    return sec > 0 ? (double(bytes) / 1e9) / sec : 0.0;
}
