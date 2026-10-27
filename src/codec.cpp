// SPDX-License-Identifier: MIT

#include "kvexp/codec.hpp"

#include "kvexp/scan.hpp"

#include <algorithm>
#include <stdexcept>

namespace kvexp {
namespace {

void append_u16(std::vector<uint8_t> &out, uint16_t v) {
    out.push_back(uint8_t(v & 0xFF));
    out.push_back(uint8_t(v >> 8));
}

int best_base(const int *hist, int width) {
    int prefix[257];
    prefix[0] = 0;
    for (int i = 0; i < 256; ++i) {
        prefix[i + 1] = prefix[i] + hist[i];
    }
    int best_c = -1;
    int best_b = 0;
    const int nbase = 256 - width + 1;
    for (int base = 0; base < nbase; ++base) {
        const int c = prefix[base + width] - prefix[base];
        if (c > best_c) {
            best_c = c;
            best_b = base;
        }
    }
    return best_b;
}

} // namespace

EncodedTensor encode_bf16(const uint16_t *kv, int seq_len, int head_dim, int block_tokens) {
    if (kv == nullptr || seq_len <= 0 || head_dim <= 0 || block_tokens <= 0) {
        throw std::invalid_argument("encode_bf16: empty input");
    }
    if (head_dim % 4 != 0 || head_dim > kMaxHeadDim) {
        throw std::invalid_argument(
            "encode_bf16: head_dim must be a multiple of 4 and at most 256");
    }
    if (block_tokens > 255) {
        throw std::invalid_argument("encode_bf16: block_tokens must fit in a byte");
    }
    if (int64_t(block_tokens) * head_dim > 65535) {
        throw std::invalid_argument("encode_bf16: block escape count would not fit in u16");
    }

    EncodedTensor enc;
    enc.seq_len = seq_len;
    enc.head_dim = head_dim;
    enc.block_tokens = block_tokens;

    std::vector<int> hist(size_t(head_dim) * 256);
    std::vector<uint8_t> base_exp(head_dim);
    std::vector<uint8_t> sm;
    std::vector<uint8_t> packed;
    std::vector<uint16_t> escapes;

    for (int t0 = 0; t0 < seq_len; t0 += block_tokens) {
        const int n = std::min(block_tokens, seq_len - t0);
        const uint16_t *src = kv + size_t(t0) * size_t(head_dim);
        std::fill(hist.begin(), hist.end(), 0);
        for (int t = 0; t < n; ++t) {
            for (int d = 0; d < head_dim; ++d) {
                ++hist[size_t(d) * 256 + bf16_exponent(src[size_t(t) * head_dim + d])];
            }
        }

        struct Choice {
            int mode;
            int bytes;
            int escapes;
        };
        Choice best{0, raw_block_bytes(n, head_dim), 0};
        for (int bits : {2, 4}) {
            const int width = (1 << bits) - 1;
            int n_esc = 0;
            for (int d = 0; d < head_dim; ++d) {
                const int b = best_base(hist.data() + size_t(d) * 256, width);
                const int *h = hist.data() + size_t(d) * 256;
                int covered = 0;
                for (int e = b; e < b + width; ++e) {
                    covered += h[e];
                }
                n_esc += n - covered;
            }
            const Choice c{bits, coded_block_bytes(n, head_dim, bits, n_esc), n_esc};
            const bool better_size = c.bytes < best.bytes;
            const bool tie = c.bytes == best.bytes;
            // Ties: prefer raw (mode 0), then fewer escapes, then narrower codes.
            const bool better_tie =
                tie && best.mode != 0 &&
                (c.escapes < best.escapes || (c.escapes == best.escapes && c.mode < best.mode));
            if (better_size || better_tie) {
                best = c;
            }
        }

        enc.bytes.push_back(uint8_t(best.mode));
        enc.bytes.push_back(uint8_t(n));
        append_u16(enc.bytes, uint16_t(best.mode == 0 ? 0 : best.escapes));
        append_u16(enc.bytes, uint16_t(head_dim));
        append_u16(enc.bytes, 0);

        if (best.mode == 0) {
            for (int i = 0; i < n * head_dim; ++i) {
                append_u16(enc.bytes, src[i]);
            }
            continue;
        }

        const int bits = best.mode;
        const int width = (1 << bits) - 1;
        const int sentinel = width;
        for (int d = 0; d < head_dim; ++d) {
            base_exp[size_t(d)] = uint8_t(best_base(hist.data() + size_t(d) * 256, width));
        }
        sm.assign(size_t(n) * size_t(head_dim), 0);
        packed.assign(size_t(exp_payload_bytes(n, head_dim, bits)), 0);
        escapes.clear();
        for (int t = 0; t < n; ++t) {
            for (int d = 0; d < head_dim; ++d) {
                const int index = t * head_dim + d;
                const uint16_t h = src[index];
                sm[size_t(index)] = bf16_sm(h);
                const int exp = bf16_exponent(h);
                const int b = base_exp[size_t(d)];
                const int delta = exp - b;
                int code;
                if (delta >= 0 && delta < width) {
                    code = delta;
                } else {
                    code = sentinel;
                    escapes.push_back(h);
                }
                if (bits == 4) {
                    if ((index & 1) == 0) {
                        packed[size_t(index >> 1)] = uint8_t(code);
                    } else {
                        packed[size_t(index >> 1)] |= uint8_t(code << 4);
                    }
                } else {
                    packed[size_t(index >> 2)] |= uint8_t(code << ((index & 3) * 2));
                }
            }
        }
        if (int(escapes.size()) != best.escapes) {
            throw std::logic_error("escape count drifted from the size model");
        }
        enc.bytes.insert(enc.bytes.end(), base_exp.begin(), base_exp.end());
        enc.bytes.insert(enc.bytes.end(), sm.begin(), sm.end());
        enc.bytes.insert(enc.bytes.end(), packed.begin(), packed.end());
        for (uint16_t h : escapes) {
            append_u16(enc.bytes, h);
        }
    }
    return enc;
}

void decode_bf16(const uint8_t *blob, size_t bytes, int seq_len, int head_dim, uint16_t *out) {
    XkvReader cur(blob, bytes, head_dim, 0);
    for (int i = 0; i < seq_len * head_dim; ++i) {
        if (cur.done()) {
            throw std::runtime_error("xkv decode ran out of blocks");
        }
        out[i] = cur.read_next();
    }
}

} // namespace kvexp
