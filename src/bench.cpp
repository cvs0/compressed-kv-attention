// SPDX-License-Identifier: MIT

#include "kvexp/attention.hpp"
#include "kvexp/codec.hpp"
#include "peak.hpp"

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <fcntl.h>
#include <fstream>
#include <iostream>
#include <omp.h>
#include <sstream>
#include <stdexcept>
#include <string>
#include <sys/mman.h>
#include <sys/stat.h>
#include <thread>
#include <unistd.h>
#include <vector>

namespace {

constexpr double kGiga = 1e9;

struct Map {
    int fd = -1;
    uint8_t *p = nullptr;
    size_t n = 0;
    ~Map() {
        if (p) {
            munmap(p, n);
        }
        if (fd >= 0) {
            close(fd);
        }
    }
    Map() = default;
    Map(const Map &) = delete;
    Map &operator=(const Map &) = delete;
    Map(Map &&o) noexcept : fd(o.fd), p(o.p), n(o.n) {
        o.fd = -1;
        o.p = nullptr;
        o.n = 0;
    }
    Map &operator=(Map &&) = delete;
};

Map map_file(const std::string &path) {
    Map m;
    m.fd = open(path.c_str(), O_RDONLY);
    if (m.fd < 0) {
        throw std::runtime_error("open failed: " + path);
    }
    struct stat st {};
    if (fstat(m.fd, &st) != 0) {
        throw std::runtime_error("fstat failed");
    }
    m.n = size_t(st.st_size);
    void *p = mmap(nullptr, m.n, PROT_READ, MAP_PRIVATE, m.fd, 0);
    if (p == MAP_FAILED) {
        throw std::runtime_error("mmap failed");
    }
    m.p = static_cast<uint8_t *>(p);
    return m;
}

uint32_t load_u32(const uint8_t *p) {
    uint32_t v;
    std::memcpy(&v, p, 4);
    return v;
}

struct Dump {
    int layers = 0;
    int kv = 0;
    int nq = 0;
    int dim = 0;
    int seq = 0;
    const uint16_t *k = nullptr;
    const uint16_t *v = nullptr;
};

Dump parse_dump(const Map &m) {
    if (m.n < 32) {
        throw std::runtime_error("dump too small");
    }
    if (std::memcmp(m.p, "KVD1", 4) != 0 || load_u32(m.p + 4) != 1) {
        throw std::runtime_error("not a KVD1 v1 dump");
    }
    Dump d;
    d.layers = int(load_u32(m.p + 8));
    d.kv = int(load_u32(m.p + 12));
    d.nq = int(load_u32(m.p + 16));
    d.dim = int(load_u32(m.p + 20));
    d.seq = int(load_u32(m.p + 24));
    const size_t plane = size_t(d.layers) * size_t(d.kv) * size_t(d.seq) * size_t(d.dim);
    const size_t need = 32 + plane * 4;
    if (m.n < need) {
        throw std::runtime_error("dump truncated");
    }
    d.k = reinterpret_cast<const uint16_t *>(m.p + 32);
    d.v = d.k + plane;
    return d;
}

struct QSnap {
    std::vector<int> pos;
    int layers = 0;
    int nq = 0;
    int dim = 0;
    std::vector<uint16_t> q; // [snap, layer, nq, dim]
};

QSnap load_qsnap(const std::string &path) {
    std::ifstream in(path, std::ios::binary);
    if (!in) {
        throw std::runtime_error("open qsnap failed");
    }
    uint32_t hdr[4];
    in.read(reinterpret_cast<char *>(hdr), sizeof(hdr));
    if (!in || std::memcmp(hdr, "QSN1", 4) != 0) {
        throw std::runtime_error("bad qsnap magic");
    }
    QSnap s;
    const int nsnap = int(hdr[1]);
    s.layers = int(hdr[2]);
    s.nq = int(hdr[3]);
    s.pos.resize(size_t(nsnap));
    in.read(reinterpret_cast<char *>(s.pos.data()), std::streamsize(nsnap * 4));
    uint32_t dim = 0;
    in.read(reinterpret_cast<char *>(&dim), 4);
    s.dim = int(dim);
    s.q.resize(size_t(nsnap) * size_t(s.layers) * size_t(s.nq) * size_t(s.dim));
    in.read(reinterpret_cast<char *>(s.q.data()), std::streamsize(s.q.size() * 2));
    if (!in) {
        throw std::runtime_error("truncated qsnap");
    }
    return s;
}

const uint16_t *head_ptr(const uint16_t *base, const Dump &d, int layer, int head) {
    const size_t plane = size_t(d.seq) * size_t(d.dim);
    return base + (size_t(layer) * size_t(d.kv) + size_t(head)) * plane;
}

const uint16_t *query_ptr(const QSnap &s, int snap, int layer, int q0) {
    const size_t layer_stride = size_t(s.nq) * size_t(s.dim);
    const size_t snap_stride = size_t(s.layers) * layer_stride;
    return s.q.data() + size_t(snap) * snap_stride + size_t(layer) * layer_stride +
           size_t(q0) * size_t(s.dim);
}

int snap_index(const QSnap &s, int seq) {
    for (int i = 0; i < int(s.pos.size()); ++i) {
        if (s.pos[size_t(i)] == seq) {
            return i;
        }
    }
    throw std::runtime_error("no query snapshot for this length");
}

uint64_t mix_floats(const float *p, int n) {
    uint64_t h = 0x9E3779B185EBCA87ull;
    for (int i = 0; i < n; ++i) {
        uint32_t u = 0;
        std::memcpy(&u, p + i, 4);
        h ^= u;
        h *= 0x100000001B3ull;
    }
    return h;
}

double median(std::vector<double> v) {
    std::sort(v.begin(), v.end());
    return v[v.size() / 2];
}

struct Timed {
    double median_ms = 0;
    double min_ms = 0;
    double max_ms = 0;
    int samples = 0;
};

template <class F> Timed time_ms(F &&fn, bool evict_between, std::vector<uint64_t> &junk) {
    fn();
    fn();
    std::vector<double> samples;
    double acc = 0;
    while (samples.size() < 5 || (acc < 350.0 && samples.size() < 25)) {
        if (evict_between) {
            uint64_t s0 = 0, s1 = 0, s2 = 0, s3 = 0;
            for (size_t i = 0; i + 3 < junk.size(); i += 4) {
                s0 += junk[i];
                s1 += junk[i + 1];
                s2 += junk[i + 2];
                s3 += junk[i + 3];
            }
            junk[0] ^= s0 ^ s1 ^ s2 ^ s3;
        }
        const auto t0 = std::chrono::steady_clock::now();
        fn();
        const auto t1 = std::chrono::steady_clock::now();
        const double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
        samples.push_back(ms);
        acc += ms;
    }
    Timed t;
    t.min_ms = *std::min_element(samples.begin(), samples.end());
    t.max_ms = *std::max_element(samples.begin(), samples.end());
    t.samples = int(samples.size());
    t.median_ms = median(std::move(samples));
    return t;
}

void append_ms(std::ostream &row, const char *name, const Timed &t) {
    row << ", \"" << name << "_ms\": " << t.median_ms << ", \"" << name << "_ms_min\": " << t.min_ms
        << ", \"" << name << "_ms_max\": " << t.max_ms << ", \"" << name
        << "_samples\": " << t.samples;
}

uint64_t stream_read(const uint64_t *p, size_t n) {
    uint64_t s0 = 0, s1 = 0, s2 = 0, s3 = 0;
    size_t i = 0;
    for (; i + 7 < n; i += 8) {
        s0 += p[i];
        s1 += p[i + 1];
        s2 += p[i + 2];
        s3 += p[i + 3];
        s0 += p[i + 4];
        s1 += p[i + 5];
        s2 += p[i + 6];
        s3 += p[i + 7];
    }
    return s0 ^ s1 ^ s2 ^ s3;
}

struct StreamPoint {
    size_t bytes = 0;
    double gbps = 0;
    int threads = 1;
    const char *method = "";
};

std::vector<StreamPoint> measure_stream(size_t max_bytes) {
    std::vector<size_t> sizes;
    for (size_t b : {8ull << 20, 16ull << 20, 32ull << 20, 64ull << 20, 96ull << 20, 128ull << 20,
                     192ull << 20, 256ull << 20, 384ull << 20, 512ull << 20}) {
        if (b <= max_bytes) {
            sizes.push_back(b);
        }
    }
    std::vector<StreamPoint> out;
    for (size_t bytes : sizes) {
        const size_t n = bytes / 8;
        uint64_t *p = static_cast<uint64_t *>(std::aligned_alloc(64, n * 8));
        if (!p) {
            break;
        }
        for (size_t i = 0; i < n; ++i) {
            p[i] = i * 0x9E3779B97F4A7C15ull;
        }
        auto sample = [&] {
            std::vector<double> ms;
            for (int r = 0; r < 7; ++r) {
                const auto t0 = std::chrono::steady_clock::now();
                const uint64_t h = stream_read(p, n);
                const auto t1 = std::chrono::steady_clock::now();
                p[0] ^= h;
                if (r >= 2) {
                    ms.push_back(std::chrono::duration<double, std::milli>(t1 - t0).count());
                }
            }
            return median(ms);
        };
        const double ms = sample();
        out.push_back(StreamPoint{bytes, (double(bytes) / kGiga) / (ms / 1e3), 1, "scalar"});
        if (kvexp::detect_cpu().avx512_bf16) {
            out.push_back(StreamPoint{bytes, avx512_read_gbps(p, bytes, 0, 1), 1, "avx512_load"});
            out.push_back(StreamPoint{bytes, avx512_read_gbps(p, bytes, 1, 1), 1, "avx512_nt"});
        }
        std::free(p);
    }
    if (kvexp::detect_cpu().avx512_bf16) {
        const size_t bytes = 512ull << 20;
        void *p = std::aligned_alloc(64, bytes);
        if (p) {
            auto *u = static_cast<uint64_t *>(p);
            for (size_t i = 0; i < bytes / 8; ++i) {
                u[i] = i * 0x9E3779B97F4A7C15ull;
            }
            for (int th : {2, 4}) {
                out.push_back(
                    StreamPoint{bytes, avx512_read_gbps(p, bytes, 0, th), th, "avx512_load"});
                out.push_back(
                    StreamPoint{bytes, avx512_read_gbps(p, bytes, 1, th), th, "avx512_nt"});
            }
            std::free(p);
        }
    }
    return out;
}

std::string cpu_model() {
    std::ifstream in("/proc/cpuinfo");
    std::string line;
    while (std::getline(in, line)) {
        if (line.rfind("model name", 0) == 0) {
            const auto c = line.find(':');
            if (c != std::string::npos) {
                return line.substr(c + 2);
            }
        }
    }
    return "unknown";
}

std::string cache_desc() {
    std::ostringstream o;
    for (int i = 0; i < 8; ++i) {
        const std::string dir = "/sys/devices/system/cpu/cpu0/cache/index" + std::to_string(i);
        std::ifstream level(dir + "/level");
        if (!level) {
            break;
        }
        std::string lv, tp, sz, shared;
        std::getline(level, lv);
        std::getline(std::ifstream(dir + "/type"), tp);
        std::getline(std::ifstream(dir + "/size"), sz);
        std::getline(std::ifstream(dir + "/shared_cpu_list"), shared);
        if (i) {
            o << "; ";
        }
        o << "L" << lv << " " << tp << " " << sz << " shared " << shared;
    }
    return o.str();
}

struct EncHead {
    kvexp::EncodedTensor k;
    kvexp::EncodedTensor v;
};

void run_head(kvexp::Backend be, const uint16_t *q, const uint16_t *k, const uint16_t *v,
              const EncHead *enc, float *out, int n_q, int dim, int seq, int threads, int block) {
    if (enc == nullptr) {
        kvexp::attention_decode(be, q, k, v, nullptr, 0, nullptr, 0, out, n_q, dim, seq, threads,
                                block);
    } else {
        kvexp::attention_decode(be, q, nullptr, nullptr, enc->k.bytes.data(), enc->k.bytes.size(),
                                enc->v.bytes.data(), enc->v.bytes.size(), out, n_q, dim, seq,
                                threads, block);
    }
}

} // namespace

int main(int argc, char **argv) {
    std::string dump_path, qsnap_path, out_path = "results/bench.json";
    bool peak_only = false;
    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        auto need = [&](const char *flag) {
            if (a != flag || i + 1 >= argc) {
                throw std::runtime_error(std::string("missing value for ") + flag);
            }
            return std::string(argv[++i]);
        };
        if (a == "--dump") {
            dump_path = need("--dump");
        } else if (a == "--qsnap") {
            qsnap_path = need("--qsnap");
        } else if (a == "--out") {
            out_path = need("--out");
        } else if (a == "--peak-only") {
            peak_only = true;
        } else {
            throw std::runtime_error("unknown arg " + a);
        }
    }
    if (!peak_only && (dump_path.empty() || qsnap_path.empty())) {
        std::cerr << "usage: kvexp_bench --dump FILE --qsnap FILE --out results/bench.json\n";
        return 2;
    }

    if (peak_only) {
        const auto cpu = kvexp::detect_cpu();
        std::cout << "cpu " << cpu_model() << "\n";
        std::cout << "cache " << cache_desc() << "\n";
        std::cout << "avx2 " << cpu.avx2 << " avx512_bf16 " << cpu.avx512_bf16 << "\n"
                  << std::flush;
        const auto stream = measure_stream(512ull << 20);
        for (const auto &s : stream) {
            std::cout << "  " << (s.bytes >> 20) << " MiB  t" << s.threads << "  " << s.method
                      << "  " << s.gbps << " GB/s\n";
        }
        if (cpu.avx2) {
            for (int th : {1, 4}) {
                std::cout << "  peak avx2_fp32_fmadd thr " << th << " " << peak_avx2_fp32_gflops(th)
                          << "\n";
            }
        }
        if (cpu.avx512_bf16) {
            for (int th : {1, 4}) {
                std::cout << "  peak avx512_fp32_fmadd thr " << th << " "
                          << peak_avx512_fp32_gflops(th) << "\n";
                std::cout << "  peak avx512_dpbf16 thr " << th << " "
                          << peak_avx512_dpbf16_gflops(th) << "\n";
            }
        }
        return 0;
    }

    const Map file = map_file(dump_path);
    const Dump dump = parse_dump(file);
    const QSnap qsnap = load_qsnap(qsnap_path);
    if (qsnap.layers != dump.layers || qsnap.nq != dump.nq || qsnap.dim != dump.dim) {
        throw std::runtime_error("qsnap does not match dump");
    }
    if (dump.nq % dump.kv != 0) {
        throw std::runtime_error("q heads not divisible by kv heads");
    }
    const int group = dump.nq / dump.kv;
    const int block = kvexp::kEncodeBlock;
    const auto cpu = kvexp::detect_cpu();
    std::vector<kvexp::Backend> backends{kvexp::Backend::Avx2};
    if (cpu.avx512_bf16) {
        backends.push_back(kvexp::Backend::Avx512Bf16);
    }

    const std::vector<int> lengths = [&] {
        std::vector<int> v;
        for (int n : {8192, 16384, 32768}) {
            if (n <= dump.seq) {
                v.push_back(n);
            }
        }
        return v;
    }();

    std::cout << "cpu " << cpu_model() << "\n";
    std::cout << "cache " << cache_desc() << "\n";
    std::cout << "avx2 " << cpu.avx2 << " avx512_bf16 " << cpu.avx512_bf16 << "\n" << std::flush;
    std::cout << "stream\n" << std::flush;
    const auto stream = measure_stream(512ull << 20);
    for (const auto &s : stream) {
        std::cout << "  " << (s.bytes >> 20) << " MiB  t" << s.threads << "  " << s.method << "  "
                  << s.gbps << " GB/s\n";
    }
    std::cout << std::flush;

    struct PeakPoint {
        const char *isa;
        int threads;
        double gflops;
        const char *flops;
    };
    std::vector<PeakPoint> peaks;
    auto add_peak = [&](const char *isa, int threads, double gflops, const char *flops) {
        peaks.push_back(PeakPoint{isa, threads, gflops, flops});
        std::cout << "  peak " << isa << " thr " << threads << " " << gflops << " GFLOP/s\n"
                  << std::flush;
    };
    if (cpu.avx2) {
        for (int th : {1, 2, 4}) {
            add_peak("avx2_fp32_fmadd", th, peak_avx2_fp32_gflops(th), "2 per fmadd lane");
        }
    }
    if (cpu.avx512_bf16) {
        for (int th : {1, 2, 4}) {
            add_peak("avx512_fp32_fmadd", th, peak_avx512_fp32_gflops(th), "2 per fmadd lane");
            add_peak("avx512_dpbf16", th, peak_avx512_dpbf16_gflops(th), "64 per vdpbf16ps");
        }
    }

    std::vector<uint64_t> junk((512ull << 20) / 8, 1);

    std::ostringstream js;
    js.setf(std::ios::fixed);
    js.precision(6);
    js << "{\n";
    js << "  \"cpu_model\": \"" << cpu_model() << "\",\n";
    js << "  \"cache\": \"" << cache_desc() << "\",\n";
    js << "  \"meminfo\": \"" << [&] {
        std::ifstream in("/proc/meminfo");
        std::string line, out;
        while (std::getline(in, line)) {
            if (line.rfind("MemTotal", 0) == 0) {
                out = line;
            }
        }
        return out;
    }() << "\",\n";
    js << "  \"n_cpus\": " << std::thread::hardware_concurrency() << ",\n";
    js << "  \"gb_definition\": \"1e9 bytes\",\n";
    js << "  \"gflops_definition\": \"1e9 flops per second\",\n";
    js << "  \"latency_ms_is\": \"median. raw_ms_min and raw_ms_max are the sample range. Two "
          "untimed calls run first. Sampling stops at 5 samples once accumulated sample time "
          "reaches 350 ms, otherwise at 25 samples.\",\n";
    js << "  \"cold_evict_bytes\": " << (512ull << 20) << ",\n";
    js << "  \"qk_pv_flops_note\": \"4 * head_dim * seq * queries. QK and PV only; exp and the "
          "horizontal add are omitted. scope=head uses one KV head and its GQA group. scope=model "
          "and scope=model_heads use every query head in every layer.\",\n";
    js << "  \"block_tokens\": " << block << ",\n";
    js << "  \"n_layers\": " << dump.layers << ",\n";
    js << "  \"n_kv_heads\": " << dump.kv << ",\n";
    js << "  \"n_q_heads\": " << dump.nq << ",\n";
    js << "  \"head_dim\": " << dump.dim << ",\n";
    js << "  \"gqa_group\": " << group << ",\n";
    js << "  \"stream_read\": [\n";
    for (size_t i = 0; i < stream.size(); ++i) {
        js << "    {\"bytes\": " << stream[i].bytes << ", \"threads\": " << stream[i].threads
           << ", \"gbps\": " << stream[i].gbps << ", \"method\": \"" << stream[i].method << "\"}";
        js << (i + 1 == stream.size() ? "\n" : ",\n");
    }
    js << "  ],\n";
    js << "  \"peak\": [\n";
    for (size_t i = 0; i < peaks.size(); ++i) {
        js << "    {\"isa\": \"" << peaks[i].isa << "\", \"threads\": " << peaks[i].threads
           << ", \"gflops\": " << peaks[i].gflops << ", \"flop_count\": \"" << peaks[i].flops
           << "\"}";
        js << (i + 1 == peaks.size() ? "\n" : ",\n");
    }
    js << "  ],\n";
    js << "  \"runs\": [\n";

    bool first = true;
    auto emit = [&](const std::string &row) {
        if (!first) {
            js << ",\n";
        }
        first = false;
        js << row;
    };

    std::ostringstream lengths_js;
    lengths_js.setf(std::ios::fixed);
    lengths_js.precision(6);
    bool first_len = true;

    auto emit_run = [&](int seq, const char *bname, int threads, int cold, const char *scope,
                        const Timed &raw_t, const Timed &xkv_t, size_t tight, size_t container_b,
                        double flops) {
        std::ostringstream row;
        row.setf(std::ios::fixed);
        row.precision(6);
        row << "    {\"seq\": " << seq << ", \"backend\": \"" << bname
            << "\", \"threads\": " << threads << ", \"cold\": " << cold << ", \"scope\": \""
            << scope << "\"";
        append_ms(row, "raw", raw_t);
        append_ms(row, "xkv", xkv_t);
        row << ", \"raw_gbps_tight\": " << ((double(tight) / kGiga) / (raw_t.median_ms / 1e3))
            << ", \"xkv_gbps_container\": "
            << ((double(container_b) / kGiga) / (xkv_t.median_ms / 1e3))
            << ", \"tight_bytes\": " << tight << ", \"container_bytes\": " << container_b
            << ", \"qk_pv_flops\": " << flops
            << ", \"raw_steps_per_s\": " << (1000.0 / raw_t.median_ms)
            << ", \"xkv_steps_per_s\": " << (1000.0 / xkv_t.median_ms) << "}";
        emit(row.str());
        std::cout << "  " << scope << " " << bname << " thr " << threads
                  << (cold ? " cold" : " hot") << " raw " << raw_t.median_ms << " ms  xkv "
                  << xkv_t.median_ms << " ms\n"
                  << std::flush;
    };

    for (int seq : lengths) {
        const int si = snap_index(qsnap, seq);
        const size_t tight_head = size_t(seq) * size_t(dump.dim) * 2u * 2u;
        const size_t tight_model = tight_head * size_t(dump.layers) * size_t(dump.kv);
        const size_t tight_layer = tight_head * size_t(dump.kv);
        const double flops_head = 4.0 * double(dump.dim) * double(seq) * double(group);
        const double flops_model =
            4.0 * double(dump.dim) * double(seq) * double(dump.nq) * double(dump.layers);
        std::vector<EncHead> enc(size_t(dump.layers) * size_t(dump.kv));
        std::vector<size_t> layer_k(size_t(dump.layers), 0);
        std::vector<size_t> layer_v(size_t(dump.layers), 0);
        size_t container = 0;
        size_t k_container = 0;
        size_t v_container = 0;
        const auto et0 = std::chrono::steady_clock::now();
        for (int layer = 0; layer < dump.layers; ++layer) {
            for (int h = 0; h < dump.kv; ++h) {
                const uint16_t *k = head_ptr(dump.k, dump, layer, h);
                const uint16_t *v = head_ptr(dump.v, dump, layer, h);
                EncHead &e = enc[size_t(layer) * dump.kv + h];
                e.k = kvexp::encode_bf16(k, seq, dump.dim, block);
                e.v = kvexp::encode_bf16(v, seq, dump.dim, block);
                layer_k[size_t(layer)] += e.k.bytes.size();
                layer_v[size_t(layer)] += e.v.bytes.size();
            }
            k_container += layer_k[size_t(layer)];
            v_container += layer_v[size_t(layer)];
            container += layer_k[size_t(layer)] + layer_v[size_t(layer)];
        }
        const double encode_s =
            std::chrono::duration<double>(std::chrono::steady_clock::now() - et0).count();
        std::cout << "seq " << seq << " ratio " << double(container) / double(tight_model) << " k "
                  << double(k_container) / double(tight_model / 2) << " v "
                  << double(v_container) / double(tight_model / 2) << " encode_s " << encode_s
                  << "\n"
                  << std::flush;

        if (!first_len) {
            lengths_js << ",\n";
        }
        first_len = false;
        lengths_js << "    {\"seq\": " << seq << ", \"encode_s\": " << encode_s
                   << ", \"tight_bytes\": " << tight_model << ", \"container_bytes\": " << container
                   << ", \"k_container_bytes\": " << k_container
                   << ", \"v_container_bytes\": " << v_container
                   << ", \"ratio\": " << double(container) / double(tight_model)
                   << ", \"k_ratio\": " << double(k_container) / double(tight_model / 2)
                   << ", \"v_ratio\": " << double(v_container) / double(tight_model / 2)
                   << ", \"layers\": [\n";
        for (int layer = 0; layer < dump.layers; ++layer) {
            const size_t kb = layer_k[size_t(layer)];
            const size_t vb = layer_v[size_t(layer)];
            const size_t half = tight_layer / 2;
            lengths_js << "      {\"layer\": " << layer << ", \"k_bytes\": " << kb
                       << ", \"v_bytes\": " << vb << ", \"tight_bytes\": " << tight_layer
                       << ", \"k_ratio\": " << double(kb) / double(half)
                       << ", \"v_ratio\": " << double(vb) / double(half)
                       << ", \"ratio\": " << double(kb + vb) / double(tight_layer) << "}";
            lengths_js << (layer + 1 == dump.layers ? "\n" : ",\n");
        }
        lengths_js << "    ]}";

        for (kvexp::Backend be : backends) {
            const char *bname = be == kvexp::Backend::Avx2 ? "avx2" : "avx512bf16";
            for (int threads : {1, 2, 4}) {
                const int layer = 0, head = 0;
                const uint16_t *q = query_ptr(qsnap, si, layer, head * group);
                const uint16_t *k = head_ptr(dump.k, dump, layer, head);
                const uint16_t *v = head_ptr(dump.v, dump, layer, head);
                const EncHead &e = enc[0];
                const size_t head_container = e.k.bytes.size() + e.v.bytes.size();
                std::vector<float> raw(size_t(group) * dump.dim), comp(raw.size());
                run_head(be, q, k, v, nullptr, raw.data(), group, dump.dim, seq, threads, block);
                run_head(be, q, k, v, &e, comp.data(), group, dump.dim, seq, threads, block);
                if (std::memcmp(raw.data(), comp.data(), raw.size() * 4) != 0) {
                    throw std::runtime_error("bit mismatch on single head");
                }
                for (int cold = 0; cold < 2; ++cold) {
                    uint64_t sink = 0;
                    const Timed raw_t = time_ms(
                        [&] {
                            run_head(be, q, k, v, nullptr, raw.data(), group, dump.dim, seq,
                                     threads, block);
                            sink ^= mix_floats(raw.data(), 8);
                        },
                        cold, junk);
                    const Timed xkv_t = time_ms(
                        [&] {
                            run_head(be, q, k, v, &e, comp.data(), group, dump.dim, seq, threads,
                                     block);
                            sink ^= mix_floats(comp.data(), 8);
                        },
                        cold, junk);
                    if (sink == 0xFFFFFFFFFFFFFFFFull) {
                        std::cout << "sink\n";
                    }
                    emit_run(seq, bname, threads, cold, "head", raw_t, xkv_t, tight_head,
                             head_container, flops_head);
                }

                std::vector<float> mraw(size_t(dump.layers) * dump.nq * dump.dim);
                std::vector<float> mcomp = mraw;
                auto invoke_model = [&](bool compressed, std::vector<float> &dst) {
                    for (int ly = 0; ly < dump.layers; ++ly) {
                        for (int h = 0; h < dump.kv; ++h) {
                            const uint16_t *qq = query_ptr(qsnap, si, ly, h * group);
                            float *o =
                                dst.data() + (size_t(ly) * dump.nq + size_t(h) * group) * dump.dim;
                            const EncHead *ee =
                                compressed ? &enc[size_t(ly) * dump.kv + h] : nullptr;
                            run_head(be, qq, head_ptr(dump.k, dump, ly, h),
                                     head_ptr(dump.v, dump, ly, h), ee, o, group, dump.dim, seq,
                                     threads, block);
                        }
                    }
                };
                invoke_model(false, mraw);
                invoke_model(true, mcomp);
                if (std::memcmp(mraw.data(), mcomp.data(), mraw.size() * 4) != 0) {
                    throw std::runtime_error("bit mismatch on full model");
                }
                for (int cold = 0; cold < 2; ++cold) {
                    uint64_t sink = 0;
                    const Timed raw_t = time_ms(
                        [&] {
                            invoke_model(false, mraw);
                            sink ^= mix_floats(mraw.data(), 16);
                        },
                        cold, junk);
                    const Timed xkv_t = time_ms(
                        [&] {
                            invoke_model(true, mcomp);
                            sink ^= mix_floats(mcomp.data(), 16);
                        },
                        cold, junk);
                    (void)sink;
                    emit_run(seq, bname, threads, cold, "model", raw_t, xkv_t, tight_model,
                             container, flops_model);
                }
            }

            const int jobs = dump.layers * dump.kv;
            for (int threads : {1, 2, 4}) {
                std::vector<float> raw(size_t(dump.layers) * dump.nq * dump.dim);
                std::vector<float> comp = raw;
                auto invoke = [&](bool compressed, std::vector<float> &dst) {
                    omp_set_dynamic(0);
#pragma omp parallel for schedule(static) num_threads(threads)
                    for (int job = 0; job < jobs; ++job) {
                        const int ly = job / dump.kv;
                        const int h = job % dump.kv;
                        const uint16_t *qq = query_ptr(qsnap, si, ly, h * group);
                        float *o =
                            dst.data() + (size_t(ly) * dump.nq + size_t(h) * group) * dump.dim;
                        const EncHead *ee = compressed ? &enc[size_t(ly) * dump.kv + h] : nullptr;
                        run_head(be, qq, head_ptr(dump.k, dump, ly, h),
                                 head_ptr(dump.v, dump, ly, h), ee, o, group, dump.dim, seq, 1,
                                 block);
                    }
                };
                invoke(false, raw);
                invoke(true, comp);
                if (std::memcmp(raw.data(), comp.data(), raw.size() * 4) != 0) {
                    throw std::runtime_error("bit mismatch on head-parallel model");
                }
                for (int cold = 0; cold < 2; ++cold) {
                    uint64_t sink = 0;
                    const Timed raw_t = time_ms(
                        [&] {
                            invoke(false, raw);
                            sink ^= mix_floats(raw.data(), 16);
                        },
                        cold, junk);
                    const Timed xkv_t = time_ms(
                        [&] {
                            invoke(true, comp);
                            sink ^= mix_floats(comp.data(), 16);
                        },
                        cold, junk);
                    (void)sink;
                    emit_run(seq, bname, threads, cold, "model_heads", raw_t, xkv_t, tight_model,
                             container, flops_model);
                }
            }
        }
    }

    js << "\n  ],\n";
    js << "  \"lengths\": [\n" << lengths_js.str() << "\n  ]\n}\n";

    std::ofstream out(out_path);
    if (!out) {
        throw std::runtime_error("cannot write " + out_path);
    }
    out << js.str();
    std::cout << "wrote " << out_path << "\n";
    return 0;
}
