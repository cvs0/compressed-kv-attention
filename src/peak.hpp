// SPDX-License-Identifier: MIT
#pragma once

#include <cstddef>

// Wall-clock peak of independent FMADD chains written as inline asm so the
// compiler cannot turn `a = fma(b, c, a)` with constant b, c into `a += k`.
// fp32: each FMADD lane counts as 2 flops.
// dpbf16: each VDPBF16PS counts as 64 flops (32 products and 32 adds into FP32).
// gflops uses 1e9 flops per second. `threads` is an OpenMP thread count.

double peak_avx2_fp32_gflops(int threads);
double peak_avx512_fp32_gflops(int threads);
double peak_avx512_dpbf16_gflops(int threads);

// `nontemporal` selects VMOVNTDQA. `threads` splits the buffer. gbps uses 1e9 bytes/s.
double avx512_read_gbps(const void *p, size_t bytes, int nontemporal, int threads);
