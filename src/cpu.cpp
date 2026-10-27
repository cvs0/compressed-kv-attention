// SPDX-License-Identifier: MIT

#include "kvexp/attention.hpp"

#include <cstdint>

namespace kvexp {
namespace {

struct Regs {
    uint32_t a, b, c, d;
};

Regs cpuid(uint32_t leaf, uint32_t sub) {
    Regs r{};
#if defined(__x86_64__)
    __asm__ volatile("cpuid" : "=a"(r.a), "=b"(r.b), "=c"(r.c), "=d"(r.d) : "a"(leaf), "c"(sub));
#else
    (void)leaf;
    (void)sub;
#endif
    return r;
}

uint64_t xgetbv0() {
    uint32_t lo = 0, hi = 0;
#if defined(__x86_64__)
    __asm__ volatile("xgetbv" : "=a"(lo), "=d"(hi) : "c"(0));
#endif
    return (uint64_t(hi) << 32) | lo;
}

} // namespace

CpuFeatures detect_cpu() {
    CpuFeatures f;
#if defined(__x86_64__)
    const Regs leaf1 = cpuid(1, 0);
    const bool osxsave = (leaf1.c & (1u << 27)) != 0;
    const bool avx = (leaf1.c & (1u << 28)) != 0;
    if (!osxsave || !avx) {
        return f;
    }
    const uint64_t xcr = xgetbv0();
    const bool ymm = (xcr & 0x6u) == 0x6u;
    const bool zmm = (xcr & 0xE6u) == 0xE6u;
    const Regs leaf7 = cpuid(7, 0);
    if (ymm && (leaf7.b & (1u << 5)) != 0) {
        f.avx2 = true;
    }
    const bool avx512f = (leaf7.b & (1u << 16)) != 0;
    const bool avx512bw = (leaf7.b & (1u << 30)) != 0;
    const bool avx512vl = (leaf7.b & (1u << 31)) != 0;
    const Regs leaf7_1 = cpuid(7, 1);
    const bool bf16 = (leaf7_1.a & (1u << 5)) != 0;
    if (zmm && avx512f && avx512bw && avx512vl && bf16) {
        f.avx512_bf16 = true;
    }
#endif
    return f;
}

} // namespace kvexp
