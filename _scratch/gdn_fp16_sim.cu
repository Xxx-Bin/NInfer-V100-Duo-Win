// C3 fp16 error-propagation simulation (2026-10-26). Host-only, driven by the real prefill dump.
//
// Question (dd-cards-20261026.md C3, revised by c3-real-check.md): with REAL operands, can a
// single static power-of-2 prescale bring T_inv / W / state into fp16, or does it need a
// per-(head,chunk) scale?  Three configurations, same math, only the quantization differs:
//
//   C  bf16 (current chunked path: W/U/v_new in bf16, state fp32, no scaling)
//   A  fp16 + GLOBAL static 2^a (scale from the measured merged maxima of the whole model)
//   B  fp16 + per-(head,chunk) static 2^a (device max-reduce produced scalar; no host sync)
//
// Reference: the same recurrence in fp64 (verbatim from gdn_w_stats.cu / gdn_w_stats_real.cu).
// Reported per configuration: max abs / max rel error of v_new and state against fp64,
// "bad" counts against the S3 acceptance (state max_rel 9.2e-4, v_new max_abs 7.8e-3),
// and the fraction of quantized entries that flush to zero.
//
// Simplification stated for the record: state_in is taken as zero at chunk 0 (the dump carries
// operands, not the incoming state); every layer restarts from zero.
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cstdio>
#include <cstdint>
#include <cmath>
#include <cstring>
#include <cstdlib>
#include <vector>
#include <algorithm>
#include <string>

static constexpr int kD     = 128;
static constexpr int kChunk = 64;
static constexpr int kLayerStride = 8;   // sample every 8th layer (8 of 64) to bound runtime

// ---------------------------------------------------------------- quantization

enum class QKind { F64, BF16, F16Global, F16Local };

struct QStat {
    long long n = 0, flushed = 0;
    double worst_scaled = 0.0;
    void note(double scaled_abs, bool is_zero_flush) {
        ++n;
        if (is_zero_flush) { ++flushed; }
        worst_scaled = std::max(worst_scaled, scaled_abs);
    }
    double flush_frac() const { return n ? double(flushed) / double(n) : 0.0; }
};

// Global scales (2^a) fixed from the measured merged maxima in c3-real-check.md:
//   K <= 1.0 -> 2^14 ; T_inv <= 1.0 -> 2^14 ; W <= 0.92 -> 2^14 ; state <= 61.2 -> 2^8
static constexpr int kScaleK = 14, kScaleTi = 14, kScaleW = 14, kScaleState = 8;

static inline double round_bf16(double x) {
    return __bfloat162float(__float2bfloat16_rn(static_cast<float>(x)));
}
static inline double quant_f16(double x, double inv_scale) {
    // value/sweep: multiply by scale, round to half, divide back
    const float scaled = static_cast<float>(x / inv_scale);   // inv_scale = 2^-a
    const __half h     = __float2half_rn(scaled);
    return __half2float(h) * inv_scale;
}
static inline bool f16_flushes(double x, double inv_scale) {
    const float scaled = static_cast<float>(std::fabs(x / inv_scale));
    return scaled < 5.9604645e-8f;   // 2^-24, smallest fp16 subnormal
}

// Quantize a whole buffer under one kind. f16_local needs the element max up front.
static void quantize(const std::vector<double>& src, std::vector<double>& dst, QKind kind,
                     int global_exp, QStat& st) {
    dst.resize(src.size());
    double max_abs = 0.0;
    for (double v : src) { max_abs = std::max(max_abs, std::fabs(v)); }
    int exp_a = global_exp;
    if (kind == QKind::F16Local) {
        exp_a = (max_abs > 0.0)
                    ? static_cast<int>(std::floor(std::log2(30000.0 / max_abs)))
                    : 0;
    }
    const double inv_scale = std::pow(2.0, -exp_a);
    for (size_t i = 0; i < src.size(); ++i) {
        const double v = src[i];
        if (kind == QKind::BF16) {
            dst[i] = round_bf16(v);
            st.note(std::fabs(v), v != 0.0 && dst[i] == 0.0);
        } else if (kind == QKind::F16Global || kind == QKind::F16Local) {
            dst[i] = quant_f16(v, inv_scale);
            st.note(std::fabs(v / inv_scale), v != 0.0 && f16_flushes(v, inv_scale));
        } else {
            dst[i] = v;
        }
    }
}

// ---------------------------------------------------------------- error metrics

struct ErrStat {
    double max_abs = 0.0;
    double max_rel = 0.0;
    long long n = 0, bad_abs = 0, bad_rel = 0;
    void add(double got, double ref, double denom, double abs_tol, double rel_tol) {
        const double a = std::fabs(got - ref);
        max_abs = std::max(max_abs, a);
        if (a > abs_tol) { ++bad_abs; }
        const double r = a / std::max(denom, 1e-30);
        max_rel = std::max(max_rel, r);
        if (r > rel_tol) { ++bad_rel; }
        ++n;
    }
    void merge(const ErrStat& o) {
        max_abs = std::max(max_abs, o.max_abs);
        max_rel = std::max(max_rel, o.max_rel);
        n += o.n; bad_abs += o.bad_abs; bad_rel += o.bad_rel;
    }
};

// ---------------------------------------------------------------- one config run

struct ConfigSpec {
    const char* name;
    QKind k, ti, w, state, vnew;
};

struct ChunkOut {
    std::vector<double> vnew;   // [kChunk][kD]
    std::vector<double> state;  // [kD][kD]
};

// Runs the chunked GDN recurrence for one (head) over all chunks of one layer.
// order: G cumsum, Af = tril(B_r exp(G_r-G_c) K K^T), T_inv recursion, W, U, v_new, state update.
static void run_head(const std::vector<double>& kn, const std::vector<double>& v,
                     const std::vector<double>& g, const std::vector<double>& beta, int T, int H_qk,
                     int H_v, int h, const ConfigSpec& spec, std::vector<ChunkOut>& out,
                     QStat& st_k, QStat& st_ti, QStat& st_w, QStat& st_state, QStat& st_vnew) {
    const int qkh = h / (H_v / H_qk);
    const int chunks = T / kChunk;
    out.assign(chunks, ChunkOut{});
    std::vector<double> S(size_t(kD) * kD, 0.0);

    std::vector<double> G(kChunk), B(kChunk);
    std::vector<double> Kc(size_t(kChunk) * kD), Vc(size_t(kChunk) * kD);
    std::vector<double> Afq(size_t(kChunk) * kChunk), Tiq(size_t(kChunk) * kChunk);
    std::vector<double> Wq(size_t(kChunk) * kD), Uq(size_t(kChunk) * kD);
    std::vector<double> Sq(size_t(kD) * kD), Vnq(size_t(kChunk) * kD);

    for (int cs = 0; cs < chunks; ++cs) {
        const int base = cs * kChunk;
        for (int t = 0; t < kChunk; ++t) {
            G[t] = g[size_t(base + t) * H_v + h];
            B[t] = beta[size_t(base + t) * H_v + h];
            if (t > 0) { G[t] += G[t - 1]; }
            for (int d = 0; d < kD; ++d) {
                Kc[size_t(t) * kD + d] = kn[(size_t(base + t) * H_qk + qkh) * kD + d];
                Vc[size_t(t) * kD + d] = v[(size_t(base + t) * H_v + h) * kD + d];
            }
        }
        // quantized views
        std::vector<double> Kq, Vq, Gq, Sq_in;
        quantize(Kc, Kq, spec.k, kScaleK, st_k);
        quantize(Vc, Vq, spec.vnew, 0, st_vnew);
        // Af in fp32 accumulate over quantized K (f16 mma path)
        for (int r = 0; r < kChunk; ++r) {
            for (int c = 0; c < kChunk; ++c) {
                double acc = 0.0;
                if (r > c) {
                    for (int d = 0; d < kD; ++d) {
                        acc += Kq[size_t(r) * kD + d] * Kq[size_t(c) * kD + d];
                    }
                    acc *= B[r] * std::exp(G[r] - G[c]);
                }
                Afq[size_t(r) * kChunk + c] = acc;
            }
        }
        // T_inv in fp32
        std::vector<double> Ti_ref(size_t(kChunk) * kChunk, 0.0);
        for (int j = 0; j < kChunk; ++j) {
            for (int i = 0; i < kChunk; ++i) {
                if (i < j) { continue; }
                double s = (i == j) ? 1.0 : 0.0;
                for (int k2 = j; k2 < i; ++k2) { s -= Afq[size_t(i) * kChunk + k2] * Ti_ref[size_t(k2) * kChunk + j]; }
                Ti_ref[size_t(i) * kChunk + j] = s;
            }
        }
        quantize(Ti_ref, Tiq, spec.ti, kScaleTi, st_ti);
        // W = T_inv @ (B exp(G) K)
        for (int t = 0; t < kChunk; ++t) {
            for (int d = 0; d < kD; ++d) {
                double s = 0.0;
                for (int j = 0; j < kChunk; ++j) {
                    s += Tiq[size_t(t) * kChunk + j] * (B[j] * std::exp(G[j]) * Kq[size_t(j) * kD + d]);
                }
                Wq[size_t(t) * kD + d] = s;
            }
        }
        std::vector<double> Wqq;
        quantize(Wq, Wqq, spec.w, kScaleW, st_w);
        // U = T_inv @ (B V)
        for (int t = 0; t < kChunk; ++t) {
            for (int d = 0; d < kD; ++d) {
                double s = 0.0;
                for (int j = 0; j < kChunk; ++j) {
                    s += Tiq[size_t(t) * kChunk + j] * (B[j] * Vq[size_t(j) * kD + d]);
                }
                Uq[size_t(t) * kD + d] = s;
            }
        }
        // state quantized view for the W @ S^T product
        quantize(S, Sq, spec.state, kScaleState, st_state);
        // v_new = U - W @ S^T
        for (int t = 0; t < kChunk; ++t) {
            for (int d = 0; d < kD; ++d) {
                double s = 0.0;
                for (int k2 = 0; k2 < kD; ++k2) { s += Wqq[size_t(t) * kD + k2] * Sq[size_t(d) * kD + k2]; }
                Vnq[size_t(t) * kD + d] = Uq[size_t(t) * kD + d] - s;
            }
        }
        std::vector<double> Vnq_q;
        quantize(Vnq, Vnq_q, spec.vnew, 0, st_vnew);
        // state update
        const double GC = G[kChunk - 1];
        for (int a = 0; a < kD; ++a) {
            for (int b = 0; b < kD; ++b) {
                double s = Sq[size_t(a) * kD + b] * std::exp(GC);
                for (int t = 0; t < kChunk; ++t) {
                    s += std::exp(GC - G[t]) * Vnq_q[size_t(t) * kD + a] * Kq[size_t(t) * kD + b];
                }
                S[size_t(a) * kD + b] = s;
            }
        }
        out[cs].vnew = Vnq_q;
        out[cs].state = S;
    }
}

// fp64 reference (no quantization at all)
static void run_head_ref(const std::vector<double>& kn, const std::vector<double>& v,
                         const std::vector<double>& g, const std::vector<double>& beta, int T,
                         int H_qk, int H_v, int h, std::vector<ChunkOut>& out) {
    const int qkh = h / (H_v / H_qk);
    const int chunks = T / kChunk;
    out.assign(chunks, ChunkOut{});
    std::vector<double> S(size_t(kD) * kD, 0.0);
    std::vector<double> G(kChunk), B(kChunk);
    std::vector<double> Af(size_t(kChunk) * kChunk), Ti(size_t(kChunk) * kChunk);
    std::vector<double> Wm(size_t(kChunk) * kD), U(size_t(kChunk) * kD);
    for (int cs = 0; cs < chunks; ++cs) {
        const int base = cs * kChunk;
        for (int t = 0; t < kChunk; ++t) {
            G[t] = g[size_t(base + t) * H_v + h];
            B[t] = beta[size_t(base + t) * H_v + h];
            if (t > 0) { G[t] += G[t - 1]; }
        }
        for (int r = 0; r < kChunk; ++r) {
            for (int c = 0; c < kChunk; ++c) {
                double val = 0.0;
                if (r > c) {
                    double kk = 0.0;
                    for (int d = 0; d < kD; ++d) {
                        kk += kn[(size_t(base + r) * H_qk + qkh) * kD + d] *
                              kn[(size_t(base + c) * H_qk + qkh) * kD + d];
                    }
                    val = B[r] * std::exp(G[r] - G[c]) * kk;
                }
                Af[size_t(r) * kChunk + c] = val;
            }
        }
        for (int j = 0; j < kChunk; ++j) {
            for (int i = 0; i < kChunk; ++i) {
                if (i < j) { Ti[size_t(i) * kChunk + j] = 0.0; continue; }
                double s = (i == j) ? 1.0 : 0.0;
                for (int k2 = j; k2 < i; ++k2) { s -= Af[size_t(i) * kChunk + k2] * Ti[size_t(k2) * kChunk + j]; }
                Ti[size_t(i) * kChunk + j] = s;
            }
        }
        for (int t = 0; t < kChunk; ++t) {
            for (int d = 0; d < kD; ++d) {
                double sw = 0.0, su = 0.0;
                for (int j = 0; j < kChunk; ++j) {
                    sw += Ti[size_t(t) * kChunk + j] * B[j] * std::exp(G[j]) *
                          kn[(size_t(base + j) * H_qk + qkh) * kD + d];
                    su += Ti[size_t(t) * kChunk + j] * B[j] *
                          v[(size_t(base + j) * H_v + h) * kD + d];
                }
                Wm[size_t(t) * kD + d] = sw;
                U[size_t(t) * kD + d]  = su;
            }
        }
        std::vector<double> Vn(size_t(kChunk) * kD);
        for (int t = 0; t < kChunk; ++t) {
            for (int d = 0; d < kD; ++d) {
                double s = U[size_t(t) * kD + d];
                for (int k2 = 0; k2 < kD; ++k2) { s -= Wm[size_t(t) * kD + k2] * S[size_t(d) * kD + k2]; }
                Vn[size_t(t) * kD + d] = s;
            }
        }
        const double GC = G[kChunk - 1];
        for (int a = 0; a < kD; ++a) {
            for (int b = 0; b < kD; ++b) {
                double s = S[size_t(a) * kD + b] * std::exp(GC);
                for (int t = 0; t < kChunk; ++t) {
                    s += std::exp(GC - G[t]) * Vn[size_t(t) * kD + a] * kn[(size_t(base + t) * H_qk + qkh) * kD + b];
                }
                S[size_t(a) * kD + b] = s;
            }
        }
        out[cs].vnew = Vn;
        out[cs].state = S;
    }
}

// ---------------------------------------------------------------- dump loading

struct Dump {
    int T = 0, H_qk = 0, H_v = 0;
    std::vector<double> k;     // [T][H_qk][D] normalized
    std::vector<double> v;     // [T][H_v][D]
    std::vector<double> g;     // [T][H_v]
    std::vector<double> beta;  // [T][H_v]
};

static bool load_dump(const std::string& path, Dump& d) {
    std::FILE* f = std::fopen(path.c_str(), "rb");
    if (f == nullptr) { return false; }
    std::int32_t hdr[8];
    if (std::fread(hdr, sizeof(hdr), 1, f) != 1) { std::fclose(f); return false; }
    if (hdr[0] != 0x47444E31 || hdr[4] != kD) { std::fclose(f); return false; }
    d.T = hdr[1]; d.H_qk = hdr[2]; d.H_v = hdr[3];
    const int norm = hdr[5];
    const size_t n_k = size_t(d.T) * d.H_qk * kD, n_v = size_t(d.T) * d.H_v * kD;
    const size_t n_g = size_t(d.T) * d.H_v;
    std::vector<__nv_bfloat16> kb(n_k), vb(n_v);
    std::vector<float> gb(n_g), bb(n_g);
    if (std::fread(kb.data(), n_k * 2, 1, f) != 1 || std::fread(vb.data(), n_v * 2, 1, f) != 1 ||
        std::fread(gb.data(), n_g * 4, 1, f) != 1 || std::fread(bb.data(), n_g * 4, 1, f) != 1) {
        std::fclose(f);
        return false;
    }
    std::fclose(f);
    d.k.resize(n_k); d.v.resize(n_v); d.g.resize(n_g); d.beta.resize(n_g);
    // Same L2 normalization as src/ops/kernel/l2norm.cuh (the dump holds the pre-normalization k).
    for (size_t row = 0; row < n_k / kD; ++row) {
        double sum = 0.0;
        for (int dd = 0; dd < kD; ++dd) {
            const double x = __bfloat162float(kb[row * kD + dd]);
            sum += x * x;
        }
        const float inv = norm ? (1.0f / std::sqrt(static_cast<float>(sum) + 1.0e-6f)) : 1.0f;
        for (int dd = 0; dd < kD; ++dd) {
            const double x = __bfloat162float(kb[row * kD + dd]) * inv;
            d.k[row * kD + dd] = norm ? __bfloat162float(__float2bfloat16_rn(static_cast<float>(x))) : x;
        }
    }
    for (size_t i = 0; i < n_v; ++i) { d.v[i] = __bfloat162float(vb[i]); }
    for (size_t i = 0; i < n_g; ++i) { d.g[i] = gb[i]; d.beta[i] = bb[i]; }
    return true;
}

int main(int argc, char** argv) {
    const char* dir = (argc > 1) ? argv[1] : "gdn-dump";
    std::vector<std::string> files;
    for (int i = 0; i < 512; i += kLayerStride) {
        char buf[1024];
        std::snprintf(buf, sizeof(buf), "%s/gdn_%03d.bin", dir, i);
        std::FILE* probe = std::fopen(buf, "rb");
        if (probe == nullptr) { continue; }
        std::fclose(probe);
        files.push_back(buf);
    }
    if (files.empty()) { std::fprintf(stderr, "no dumps under %s\n", dir); return 1; }

    const ConfigSpec specs[] = {
        // C is the REAL current path: T_inv stays fp32 in smem (never crosses HBM), W/U/v_new
        // are bf16 in the workspace, state is fp32.
        {"C bf16 现状(Ti fp32)", QKind::BF16, QKind::F64, QKind::BF16, QKind::F64, QKind::BF16},
        {"A f16 全局静态      ", QKind::F16Global, QKind::F16Global, QKind::F16Global, QKind::F16Global, QKind::BF16},
        {"B f16 per-chunk     ", QKind::F16Global, QKind::F16Local, QKind::F16Local, QKind::F16Local, QKind::BF16},
        {"A0 f16 W/S, Ti fp32 ", QKind::F16Global, QKind::F64, QKind::F16Global, QKind::F16Global, QKind::BF16},
    };
    constexpr int kNumSpecs = sizeof(specs) / sizeof(specs[0]);

    printf("== C3 fp16 error propagation, real prefill dump (%zu of 64 layers, stride %d) ==\n",
           files.size(), kLayerStride);
    printf("   tolerance (S3 measured): state max_rel 9.2e-4 ; v_new max_abs 7.8e-3\n");

    ErrStat e_vnew[4], e_state[4];
    QStat q_ti[4], q_w[4], q_state[4];
    for (size_t fi = 0; fi < files.size(); ++fi) {
        Dump d;
        if (!load_dump(files[fi], d)) { std::fprintf(stderr, "skip %s\n", files[fi].c_str()); continue; }
        if (d.T % kChunk != 0) { continue; }
        for (int h = 0; h < d.H_v; ++h) {
            std::vector<ChunkOut> ref;
            run_head_ref(d.k, d.v, d.g, d.beta, d.T, d.H_qk, d.H_v, h, ref);
            for (int si = 0; si < kNumSpecs; ++si) {
                std::vector<ChunkOut> got;
                QStat st_k, st_vn;
                run_head(d.k, d.v, d.g, d.beta, d.T, d.H_qk, d.H_v, h, specs[si], got, st_k,
                         q_ti[si], q_w[si], q_state[si], st_vn);
                for (size_t c = 0; c < ref.size(); ++c) {
                    // S3-style normalization: relative error is measured against the largest
                    // magnitude of the reference tensor in this chunk (not per element, which is
                    // meaningless for the deep-decay dust).
                    double vscale = 0.0, sscale = 0.0;
                    for (double x : ref[c].vnew) { vscale = std::max(vscale, std::fabs(x)); }
                    for (double x : ref[c].state) { sscale = std::max(sscale, std::fabs(x)); }
                    for (size_t i = 0; i < ref[c].vnew.size(); ++i) {
                        e_vnew[si].add(got[c].vnew[i], ref[c].vnew[i], vscale, 7.8e-3, 9.2e-4);
                    }
                    for (size_t i = 0; i < ref[c].state.size(); ++i) {
                        e_state[si].add(got[c].state[i], ref[c].state[i], sscale, 7.8e-3, 9.2e-4);
                    }
                }
            }
        }
        printf("  layer %03d done\n", int(fi) * kLayerStride);
        std::fflush(stdout);
    }

    printf("\n-- summary (vs fp64 reference) --\n");
    for (int si = 0; si < kNumSpecs; ++si) {
        printf("%s\n", specs[si].name);
        printf("   v_new : n=%lld max_abs=%.3e max_rel=%.3e bad(abs>7.8e-3)=%lld bad(rel>9.2e-4)=%lld\n",
               e_vnew[si].n, e_vnew[si].max_abs, e_vnew[si].max_rel, e_vnew[si].bad_abs,
               e_vnew[si].bad_rel);
        printf("   state : n=%lld max_abs=%.3e max_rel=%.3e bad(abs>7.8e-3)=%lld bad(rel>9.2e-4)=%lld\n",
               e_state[si].n, e_state[si].max_abs, e_state[si].max_rel, e_state[si].bad_abs,
               e_state[si].bad_rel);
        printf("   flush : T_inv=%.4f%% W=%.4f%% state=%.4f%%\n", 100.0 * q_ti[si].flush_frac(),
               100.0 * q_w[si].flush_frac(), 100.0 * q_state[si].flush_frac());
    }
    printf("GDN_FP16_SIM_DONE\n");
    return 0;
}
