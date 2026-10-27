// SPDX-License-Identifier: MIT

#include "kvexp/attention.hpp"
#include "kvexp/codec.hpp"

#include <cmath>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <string>
#include <vector>

namespace {

int g_fails = 0;

void check(bool cond, const char *expr, const char *file, int line) {
    if (!cond) {
        std::cerr << "FAIL " << file << ":" << line << " " << expr << "\n";
        ++g_fails;
    }
}

#define CHECK(cond) ::check(static_cast<bool>(cond), #cond, __FILE__, __LINE__)

uint32_t rng_state = 0x12345678u;
uint32_t rnd() {
    rng_state ^= rng_state << 13;
    rng_state ^= rng_state >> 17;
    rng_state ^= rng_state << 5;
    return rng_state;
}

uint16_t make_bf16(int exp, int mant, int sign) {
    return uint16_t((sign ? 0x8000 : 0) | ((exp & 0xFF) << 7) | (mant & 0x7F));
}

std::vector<uint16_t> clustered(int n, int base, int spread) {
    std::vector<uint16_t> v(n);
    for (int i = 0; i < n; ++i) {
        int exp = base + int(rnd() % uint32_t(spread));
        if ((rnd() % 200u) == 0) {
            exp = (base + 80) & 0xFF;
        }
        v[size_t(i)] = make_bf16(exp, int(rnd() & 0x7F), int(rnd() & 1));
    }
    return v;
}

bool bits_equal(const float *a, const float *b, int n) {
    return std::memcmp(a, b, size_t(n) * sizeof(float)) == 0;
}

void test_roundtrip() {
    const int dim = 64;
    const int seq = 300;
    const int blocks[] = {16, 32, 64, 128};
    for (int block : blocks) {
        auto kv = clustered(seq * dim, 110, 6);
        // A run of exact zeros and a subnormal.
        kv[0] = 0;
        kv[1] = 0x8000;
        kv[2] = 0x0001;
        kv[3] = 0x7F80; // inf
        kv[4] = 0x7FC1; // nan
        auto enc = kvexp::encode_bf16(kv.data(), seq, dim, block);
        std::vector<uint16_t> got(kv.size());
        kvexp::decode_bf16(enc.bytes.data(), enc.bytes.size(), seq, dim, got.data());
        CHECK(got == kv);
        CHECK(!enc.bytes.empty());
    }
}

void test_size_formula() {
    const int dim = 64;
    const int seq = 128;
    auto kv = clustered(seq * dim, 100, 3);
    auto enc = kvexp::encode_bf16(kv.data(), seq, dim, 128);
    // One block. Container is either raw or coded, and decode matches.
    const int raw = kvexp::raw_block_bytes(seq, dim);
    const int c4 = kvexp::coded_block_bytes(seq, dim, 4, 0);
    CHECK(int(enc.bytes.size()) == raw || int(enc.bytes.size()) >= c4 - 64 * seq);
    std::vector<uint16_t> got(kv.size());
    kvexp::decode_bf16(enc.bytes.data(), enc.bytes.size(), seq, dim, got.data());
    CHECK(got == kv);
}

void attn(kvexp::Backend be, const std::vector<uint16_t> &q, const std::vector<uint16_t> &k,
          const std::vector<uint16_t> &v, const kvexp::EncodedTensor *ek,
          const kvexp::EncodedTensor *ev, std::vector<float> &out, int n_q, int dim, int seq,
          int threads, int split) {
    const uint16_t *kr = ek ? nullptr : k.data();
    const uint16_t *vr = ev ? nullptr : v.data();
    const uint8_t *kx = ek ? ek->bytes.data() : nullptr;
    const uint8_t *vx = ev ? ev->bytes.data() : nullptr;
    const size_t kb = ek ? ek->bytes.size() : 0;
    const size_t vb = ev ? ev->bytes.size() : 0;
    kvexp::attention_decode(be, q.data(), kr, vr, kx, kb, vx, vb, out.data(), n_q, dim, seq,
                            threads, split);
}

void test_identity_token() {
    const int dim = 8;
    const int n_q = 1;
    const int seq = 1;
    std::vector<uint16_t> q(dim, 0x3F80);
    std::vector<uint16_t> k = q;
    std::vector<uint16_t> v(dim, 0x3F80);
    v[0] = 0x4000; // bf16 2.0
    std::vector<float> out(dim);
    attn(kvexp::Backend::Scalar, q, k, v, nullptr, nullptr, out, n_q, dim, seq, 1, 128);
    CHECK(out[0] == kvexp::bf16_to_fp32(v[0]));
    for (int d = 1; d < dim; ++d) {
        CHECK(out[d] == 1.0f);
    }
    auto cpu = kvexp::detect_cpu();
    if (cpu.avx2) {
        std::vector<float> o2(dim);
        attn(kvexp::Backend::Avx2, q, k, v, nullptr, nullptr, o2, n_q, dim, seq, 1, 128);
        CHECK(bits_equal(out.data(), o2.data(), dim));
    }
}

void test_bitexact(int dim, int n_q, int seq, int block, int threads) {
    auto cpu = kvexp::detect_cpu();
    auto k = clustered(seq * dim, 108, 8);
    auto v = clustered(seq * dim, 112, 5);
    std::vector<uint16_t> q(size_t(n_q) * size_t(dim));
    for (size_t i = 0; i < q.size(); ++i) {
        q[i] = make_bf16(120, int(rnd() & 0x7F), int(rnd() & 1));
    }
    auto ek = kvexp::encode_bf16(k.data(), seq, dim, block);
    auto ev = kvexp::encode_bf16(v.data(), seq, dim, block);
    const int nout = n_q * dim;

    std::vector<float> s_raw(nout), s_x(nout);
    attn(kvexp::Backend::Scalar, q, k, v, nullptr, nullptr, s_raw, n_q, dim, seq, threads, block);
    attn(kvexp::Backend::Scalar, q, k, v, &ek, &ev, s_x, n_q, dim, seq, threads, block);
    CHECK(bits_equal(s_raw.data(), s_x.data(), nout));

    if (cpu.avx2 && dim % 8 == 0) {
        std::vector<float> a_raw(nout), a_x(nout);
        attn(kvexp::Backend::Avx2, q, k, v, nullptr, nullptr, a_raw, n_q, dim, seq, threads, block);
        attn(kvexp::Backend::Avx2, q, k, v, &ek, &ev, a_x, n_q, dim, seq, threads, block);
        CHECK(bits_equal(a_raw.data(), a_x.data(), nout));
        if (threads == 1) {
            CHECK(bits_equal(s_raw.data(), a_raw.data(), nout));
        }
    }
    if (cpu.avx512_bf16 && dim % 32 == 0) {
        std::vector<float> z_raw(nout), z_x(nout);
        attn(kvexp::Backend::Avx512Bf16, q, k, v, nullptr, nullptr, z_raw, n_q, dim, seq, threads,
             block);
        attn(kvexp::Backend::Avx512Bf16, q, k, v, &ek, &ev, z_x, n_q, dim, seq, threads, block);
        CHECK(bits_equal(z_raw.data(), z_x.data(), nout));
    }
}

} // namespace

int main() {
    test_roundtrip();
    test_size_formula();
    test_identity_token();
    test_bitexact(64, 1, 1, 128, 1);
    test_bitexact(64, 7, 17, 128, 1);
    test_bitexact(64, 7, 128, 128, 1);
    test_bitexact(64, 3, 200, 64, 1);
    test_bitexact(64, 7, 300, 64, 4);
    test_bitexact(32, 2, 90, 32, 1);
    test_bitexact(16, 1, 40, 16, 1);
    test_bitexact(64, 7, 256, 128, 4);
    if (g_fails != 0) {
        std::cerr << g_fails << " checks failed\n";
        return 1;
    }
    std::cout << "ok\n";
    return 0;
}
