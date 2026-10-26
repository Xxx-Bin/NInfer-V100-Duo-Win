// C3 real-data checkpoint (2026-10-26): recheck the synthetic W/T_inv fp16 exponent sweep
// (_scratch/gdn_w_stats.cu, dd-cards-20261026.md C3) against REAL prefill-chunk operands.
//
// Input: dumps written by the env-gated hook in src/ops/linear_attention/gated_delta_net/
// gated_delta_net.cpp (NINFER_GDN_DUMP=<dir>). One file per GDN invocation (one per layer),
// header = int32[8] {magic, T, H_qk, H_v, D, normalize_qk, 0, 0}, then raw
//   k    bf16 [T][H_qk][D]
//   v    bf16 [T][H_v ][D]
//   g    fp32 [T][H_v ]
//   beta fp32 [T][H_v ]
//
// The math is a verbatim copy of the fp64 host oracle in gdn_w_stats.cu (same Af, T_inv
// recursion, W, U, v_new, state recursion, same Stats/report format), so "synthetic vs real"
// is a same-code comparison. Differs only in the input source: instead of generating
// synthetic k/g/beta it applies the pipeline's own L2 normalization to the dumped k
// (bf16 -> fp32 sum -> rsqrt(sum+1e-6) -> bf16, matching src/ops/kernel/l2norm.cuh) and
// then runs the identical chunk recurrence over the real tokens.
#include <cuda_bf16.h>
#include <cstdio>
#include <cstdint>
#include <cmath>
#include <cstring>
#include <cstdlib>
#include <vector>
#include <algorithm>
#include <string>

static constexpr int kD     = 128;   // asserted against the dump header
static constexpr int kChunk = 64;

struct Stats {
    long long n = 0, n_zero = 0;
    double max_abs = 0.0, min_nonzero = 1e300;
    long long n_sub16 = 0;
    void add(double v) {
        ++n;
        const double a = std::fabs(v);
        if (a == 0.0) { ++n_zero; return; }
        if (a > max_abs) max_abs = a;
        if (a < min_nonzero) min_nonzero = a;
        const int e = static_cast<int>(std::floor(std::log2(a)));
        if (e < -14) ++n_sub16;
    }
    void merge(const Stats& o) {
        n += o.n; n_zero += o.n_zero; n_sub16 += o.n_sub16;
        max_abs = std::max(max_abs, o.max_abs);
        min_nonzero = std::min(min_nonzero, o.min_nonzero);
    }
    double span() const { return (min_nonzero > 0.0 && max_abs > 0.0)
                                     ? std::log2(max_abs / min_nonzero) : 0.0; }
};

static void report(const char* tag, const Stats& s) {
    if (s.n == 0 || s.max_abs == 0.0) {
        printf("  %-7s n=%lld all-zero\n", tag, s.n);
        return;
    }
    const double span = s.span();
    const int a = static_cast<int>(std::floor(std::log2(30000.0 / s.max_abs)));
    const double scaled_max = s.max_abs * std::pow(2.0, a);
    const double scaled_min = s.min_nonzero * std::pow(2.0, a);
    const double sub_frac = s.n ? double(s.n_sub16) / double(s.n) : 0.0;
    printf("  %-7s n=%lld max=%.3e minNZ=%.3e span=%.1fbits scale=2^%d "
           "scaled[%.2e, %.2e] subnormal=%.4f%%\n",
           tag, s.n, s.max_abs, s.min_nonzero, span, a, scaled_min, scaled_max, 100.0 * sub_frac);
}

struct LayerStats {
    Stats af, expg, ti, w, u, vn, st;
    double ti_max_lo = 1e300, ti_max_hi = 0.0;
    double st_max_lo = 1e300, st_max_hi = 0.0;
};

// One dump file -> stats for every (value head, chunk) inside it.
static bool analyze_file(const std::string& path, LayerStats& out, int& T_out, int& hqk_out,
                         int& hv_out, int& norm_out) {
    std::FILE* f = std::fopen(path.c_str(), "rb");
    if (f == nullptr) { fprintf(stderr, "cannot open %s\n", path.c_str()); return false; }
    std::int32_t hdr[8];
    if (std::fread(hdr, sizeof(hdr), 1, f) != 1) { std::fclose(f); return false; }
    const std::int32_t T = hdr[1], H_qk = hdr[2], H_v = hdr[3], D = hdr[4], norm = hdr[5];
    if (hdr[0] != 0x47444E31 || D != kD || T <= 0 || (T % kChunk) != 0 || H_qk < 1 || H_v < 1) {
        fprintf(stderr, "%s: bad header (magic=%08x T=%d H_qk=%d H_v=%d D=%d)\n", path.c_str(),
                hdr[0], T, H_qk, H_v, D);
        std::fclose(f);
        return false;
    }
    const size_t n_k = size_t(T) * H_qk * D, n_v = size_t(T) * H_v * D, n_g = size_t(T) * H_v;
    std::vector<__nv_bfloat16> k(n_k), v(n_v);
    std::vector<float> g(n_g), beta(n_g);
    if (std::fread(k.data(), n_k * 2, 1, f) != 1 || std::fread(v.data(), n_v * 2, 1, f) != 1 ||
        std::fread(g.data(), n_g * 4, 1, f) != 1 ||
        std::fread(beta.data(), n_g * 4, 1, f) != 1) {
        fprintf(stderr, "%s: short read\n", path.c_str());
        std::fclose(f);
        return false;
    }
    std::fclose(f);

    // Pipeline L2 normalization (src/ops/kernel/l2norm.cuh): fp32 sum of the bf16 squares,
    // inv = rsqrt(sum + 1e-6), product rounded back to bf16.
    std::vector<double> kn(n_k);
    for (size_t row = 0; row < n_k / D; ++row) {
        double sum = 0.0;
        for (int d = 0; d < D; ++d) {
            const double x = __bfloat162float(k[row * D + d]);
            sum += x * x;
        }
        const float inv = norm ? (1.0f / std::sqrt(static_cast<float>(sum) + 1.0e-6f)) : 1.0f;
        for (int d = 0; d < D; ++d) {
            const double x = __bfloat162float(k[row * D + d]) * inv;
            kn[row * D + d] = norm ? __bfloat162float(__float2bfloat16_rn(static_cast<float>(x)))
                                   : x;
        }
    }

    for (int h = 0; h < H_v; ++h) {
        const int qkh = h / (H_v / H_qk);
        std::vector<double> S(size_t(D) * D, 0.0);
        std::vector<double> G(kChunk), B(kChunk), Af(size_t(kChunk) * kChunk),
            Ti(size_t(kChunk) * kChunk), Wm(size_t(kChunk) * D), Vn(size_t(kChunk) * D);
        for (int cs = 0; cs < T; cs += kChunk) {
            for (int t = 0; t < kChunk; ++t) {
                G[t] = g[size_t(cs + t) * H_v + h];
                B[t] = beta[size_t(cs + t) * H_v + h];
                if (t > 0) { G[t] += G[t - 1]; }
                out.expg.add(std::exp(G[t]));
            }
            for (int r = 0; r < kChunk; ++r) {
                for (int c = 0; c < kChunk; ++c) {
                    double val = 0.0;
                    if (r > c) {
                        double kk = 0.0;
                        for (int d = 0; d < D; ++d) {
                            kk += kn[(size_t(cs + r) * H_qk + qkh) * D + d] *
                                  kn[(size_t(cs + c) * H_qk + qkh) * D + d];
                        }
                        val = B[r] * std::exp(G[r] - G[c]) * kk;
                        out.af.add(val);
                    }
                    Af[size_t(r) * kChunk + c] = val;
                }
            }
            for (int j = 0; j < kChunk; ++j) {
                for (int i = 0; i < kChunk; ++i) {
                    if (i < j) { Ti[size_t(i) * kChunk + j] = 0.0; continue; }
                    double s = (i == j) ? 1.0 : 0.0;
                    for (int k2 = j; k2 < i; ++k2) {
                        s -= Af[size_t(i) * kChunk + k2] * Ti[size_t(k2) * kChunk + j];
                    }
                    Ti[size_t(i) * kChunk + j] = s;
                    out.ti.add(s);
                }
            }
            double ti_max = 0.0;
            for (int i = 0; i < kChunk; ++i) {
                for (int j = 0; j <= i; ++j) {
                    ti_max = std::max(ti_max, std::fabs(Ti[size_t(i) * kChunk + j]));
                }
            }
            out.ti_max_lo = std::min(out.ti_max_lo, ti_max);
            out.ti_max_hi = std::max(out.ti_max_hi, ti_max);
            for (int t = 0; t < kChunk; ++t) {
                for (int d = 0; d < D; ++d) {
                    double sw = 0.0;
                    for (int s = 0; s < kChunk; ++s) {
                        sw += Ti[size_t(t) * kChunk + s] * B[s] * std::exp(G[s]) *
                              kn[(size_t(cs + s) * H_qk + qkh) * D + d];
                    }
                    Wm[size_t(t) * D + d] = sw;
                    out.w.add(sw);
                }
            }
            for (int t = 0; t < kChunk; ++t) {
                for (int d = 0; d < D; ++d) {
                    double su = 0.0;
                    for (int s = 0; s < kChunk; ++s) {
                        su += Ti[size_t(t) * kChunk + s] * B[s] *
                              __bfloat162float(v[(size_t(cs + s) * H_v + h) * D + d]);
                    }
                    out.u.add(su);
                    double s2 = su;
                    for (int k2 = 0; k2 < D; ++k2) {
                        s2 -= Wm[size_t(t) * D + k2] * S[size_t(d) * D + k2];
                    }
                    Vn[size_t(t) * D + d] = s2;
                    out.vn.add(s2);
                }
            }
            const double GC = G[kChunk - 1];
            for (int a = 0; a < D; ++a) {
                for (int b = 0; b < D; ++b) {
                    double s = S[size_t(a) * D + b] * std::exp(GC);
                    for (int t = 0; t < kChunk; ++t) {
                        s += std::exp(GC - G[t]) * Vn[size_t(t) * D + a] *
                             kn[(size_t(cs + t) * H_qk + qkh) * D + b];
                    }
                    S[size_t(a) * D + b] = s;
                }
            }
            double smax = 0.0;
            for (int a = 0; a < D; ++a) {
                for (int b = 0; b < D; ++b) {
                    const double val = S[size_t(a) * D + b];
                    smax = std::max(smax, std::fabs(val));
                    out.st.add(val);
                }
            }
            out.st_max_lo = std::min(out.st_max_lo, smax);
            out.st_max_hi = std::max(out.st_max_hi, smax);
        }
    }
    T_out = T; hqk_out = H_qk; hv_out = H_v; norm_out = norm;
    return true;
}

int main(int argc, char** argv) {
    const char* dir = (argc > 1) ? argv[1] : ".";
    std::vector<std::string> files;
    for (int i = 0; i < 512; ++i) {
        char buf[1024];
        std::snprintf(buf, sizeof(buf), "%s/gdn_%03d.bin", dir, i);
        std::FILE* probe = std::fopen(buf, "rb");
        if (probe == nullptr) { continue; }
        std::fclose(probe);
        files.push_back(buf);
    }
    if (files.empty()) {
        fprintf(stderr, "no dumps under %s\n", dir);
        return 1;
    }
    printf("== real prefill chunk (%zu GDN invocations) ==\n", files.size());
    LayerStats all;
    double worst_ti_span = 0.0, worst_w_span = 0.0, worst_st_span = 0.0;
    int worst_ti_idx = -1, worst_w_idx = -1, worst_st_idx = -1;
    for (size_t i = 0; i < files.size(); ++i) {
        LayerStats ls;
        int T = 0, hqk = 0, hv = 0, norm = 0;
        if (!analyze_file(files[i], ls, T, hqk, hv, norm)) { continue; }
        if (i == 0) { printf("  T=%d H_qk=%d H_v=%d normalize_qk=%d\n", T, hqk, hv, norm); }
        const double tis = ls.ti.span(), ws = ls.w.span(), sts = ls.st.span();
        if (tis > worst_ti_span) { worst_ti_span = tis; worst_ti_idx = int(i); }
        if (ws > worst_w_span) { worst_w_span = ws; worst_w_idx = int(i); }
        if (sts > worst_st_span) { worst_st_span = sts; worst_st_idx = int(i); }
        printf("  layer %03zu  |Ti|max=[%.3f, %.3f] |S|max=[%.3f, %.3f] "
               "T_inv span=%.1f W span=%.1f state span=%.1f\n",
               i, ls.ti_max_lo, ls.ti_max_hi, ls.st_max_lo, ls.st_max_hi, tis, ws, sts);
        all.ti.merge(ls.ti); all.w.merge(ls.w); all.st.merge(ls.st);
        all.af.merge(ls.af); all.expg.merge(ls.expg); all.u.merge(ls.u); all.vn.merge(ls.vn);
        all.ti_max_lo = std::min(all.ti_max_lo, ls.ti_max_lo);
        all.ti_max_hi = std::max(all.ti_max_hi, ls.ti_max_hi);
        all.st_max_lo = std::min(all.st_max_lo, ls.st_max_lo);
        all.st_max_hi = std::max(all.st_max_hi, ls.st_max_hi);
    }
    printf("-- all layers merged --\n");
    printf("  chunk |Ti|max: [%.3f, %.3f]   state |S|max: [%.3f, %.3f]\n", all.ti_max_lo,
           all.ti_max_hi, all.st_max_lo, all.st_max_hi);
    report("Af", all.af);
    report("expG", all.expg);
    report("T_inv", all.ti);
    report("W", all.w);
    report("U", all.u);
    report("v_new", all.vn);
    report("state", all.st);
    printf("  worst spans: T_inv %.1f bits (layer %d) / W %.1f bits (layer %d) / "
           "state %.1f bits (layer %d); fp16 normal domain = 29 bits\n",
           worst_ti_span, worst_ti_idx, worst_w_span, worst_w_idx, worst_st_span, worst_st_idx);
    printf("GDN_W_STATS_REAL_DONE\n");
    return 0;
}
