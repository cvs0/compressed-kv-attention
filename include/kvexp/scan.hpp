// SPDX-License-Identifier: MIT
#pragma once

#include "kvexp/bf16.hpp"

#include <cstddef>
#include <cstdint>
#include <stdexcept>

namespace kvexp {

struct BlockView {
    int mode = 0;
    int n_tokens = 0;
    int n_escape = 0;
    const uint8_t *base = nullptr;
    const uint8_t *sm = nullptr;
    const uint8_t *exp = nullptr;
    const uint16_t *escapes = nullptr;
    const uint16_t *raw = nullptr;
    const uint8_t *next = nullptr;
};

inline int u16_le(const uint8_t *p) {
    return int(p[0]) | (int(p[1]) << 8);
}

inline BlockView parse_block(const uint8_t *p, const uint8_t *end, int dim) {
    if (end - p < 8) {
        throw std::runtime_error("truncated xkv block header");
    }
    BlockView b;
    b.mode = p[0];
    b.n_tokens = p[1];
    b.n_escape = u16_le(p + 2);
    int hdim = u16_le(p + 4);
    if (hdim != dim || b.n_tokens <= 0) {
        throw std::runtime_error("xkv block dimension or length mismatch");
    }
    p += 8;
    if (b.mode == 0) {
        const size_t nbytes = size_t(b.n_tokens) * size_t(dim) * 2u;
        if (size_t(end - p) < nbytes) {
            throw std::runtime_error("truncated raw xkv block");
        }
        b.raw = reinterpret_cast<const uint16_t *>(p);
        b.next = p + nbytes;
        return b;
    }
    if (b.mode != 2 && b.mode != 4) {
        throw std::runtime_error("unknown xkv block mode");
    }
    const size_t sm_bytes = size_t(b.n_tokens) * size_t(dim);
    const size_t exp_bytes = size_t(exp_payload_bytes(b.n_tokens, dim, b.mode));
    const size_t esc_bytes = size_t(b.n_escape) * 2u;
    const size_t need = size_t(dim) + sm_bytes + exp_bytes + esc_bytes;
    if (size_t(end - p) < need) {
        throw std::runtime_error("truncated coded xkv block");
    }
    b.base = p;
    b.sm = p + dim;
    b.exp = b.sm + sm_bytes;
    b.escapes = reinterpret_cast<const uint16_t *>(b.exp + exp_bytes);
    b.next = b.exp + exp_bytes + esc_bytes;
    return b;
}

inline int residual_code(const uint8_t *exp, int index, int bits) {
    if (bits == 4) {
        uint8_t byte = exp[index >> 1];
        return (index & 1) ? (byte >> 4) : (byte & 0x0F);
    }
    uint8_t byte = exp[index >> 2];
    return (byte >> ((index & 3) * 2)) & 0x03;
}

// Reader positioned at a block-aligned token. read_next() walks one BF16 in
// scan order and consumes the escape stream.
class XkvReader {
public:
    XkvReader() = default;

    XkvReader(const uint8_t *data, size_t size, int dim, int token) {
        seek(data, size, dim, token);
    }

    int mode() const { return blk_.mode; }
    int local() const { return local_; }
    int n_tokens() const { return blk_.n_tokens; }
    const BlockView &block() const { return blk_; }
    const uint16_t *&escape() { return esc_; }
    bool done() const { return done_; }

    void seek(const uint8_t *data, size_t size, int dim, int token) {
        dim_ = dim;
        end_ = data + size;
        const uint8_t *p = data;
        int pos = 0;
        if (token < 0) {
            throw std::runtime_error("negative token");
        }
        while (pos < token) {
            if (p >= end_) {
                throw std::runtime_error("xkv seek past end");
            }
            blk_ = parse_block(p, end_, dim_);
            if (pos + blk_.n_tokens > token) {
                throw std::runtime_error("token split is not aligned to an encoded block");
            }
            pos += blk_.n_tokens;
            p = blk_.next;
        }
        if (p >= end_) {
            done_ = true;
            local_ = 0;
            return;
        }
        blk_ = parse_block(p, end_, dim_);
        local_ = 0;
        di_ = 0;
        esc_ = blk_.escapes;
        done_ = false;
    }

    uint16_t read_next() {
        uint16_t value;
        if (blk_.mode == 0) {
            value = blk_.raw[local_ * dim_ + di_];
        } else {
            const int index = local_ * dim_ + di_;
            const uint8_t sm = blk_.sm[index];
            const int code = residual_code(blk_.exp, index, blk_.mode);
            const int sentinel = (1 << blk_.mode) - 1;
            if (code == sentinel) {
                value = *esc_++;
            } else {
                value = bf16_from_delta(blk_.base[di_], uint8_t(code), sm);
            }
        }
        ++di_;
        if (di_ == dim_) {
            di_ = 0;
            ++local_;
            if (local_ == blk_.n_tokens) {
                if (blk_.next >= end_) {
                    done_ = true;
                } else {
                    blk_ = parse_block(blk_.next, end_, dim_);
                    local_ = 0;
                    esc_ = blk_.escapes;
                }
            }
        }
        return value;
    }

private:
    BlockView blk_{};
    const uint8_t *end_ = nullptr;
    const uint16_t *esc_ = nullptr;
    int dim_ = 0;
    int local_ = 0;
    int di_ = 0;
    bool done_ = true;
};

// SIMD kernels walk whole tokens from a block-aligned position.
struct XkvPos {
    BlockView blk{};
    const uint8_t *end = nullptr;
    const uint16_t *esc = nullptr;
    int dim = 0;
    int local = 0;
    bool done = true;

    void advance_token() {
        ++local;
        if (local < blk.n_tokens) {
            return;
        }
        if (blk.next >= end) {
            done = true;
            return;
        }
        blk = parse_block(blk.next, end, dim);
        local = 0;
        esc = blk.escapes;
        done = false;
    }
};

inline XkvPos open_xkv(const uint8_t *data, size_t size, int dim, int token) {
    XkvPos pos;
    pos.end = data + size;
    pos.dim = dim;
    const uint8_t *p = data;
    int at = 0;
    while (at < token) {
        if (p >= pos.end) {
            throw std::runtime_error("xkv seek past end");
        }
        BlockView b = parse_block(p, pos.end, dim);
        if (at + b.n_tokens > token) {
            throw std::runtime_error("token split is not aligned to an encoded block");
        }
        at += b.n_tokens;
        p = b.next;
    }
    if (p >= pos.end) {
        pos.done = true;
        return pos;
    }
    pos.blk = parse_block(p, pos.end, dim);
    pos.local = 0;
    pos.esc = pos.blk.escapes;
    pos.done = false;
    return pos;
}

} // namespace kvexp
