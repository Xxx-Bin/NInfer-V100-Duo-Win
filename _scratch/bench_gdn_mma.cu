// S5-0 gate microbenchmark (2026-10-26): on sm_70, how much faster is a single f16 product
// than the two emulation layers S4 actually ran?
//
// Unit of work for every variant: one logical m16n8k16 tile-MAC (16x8x16 = 2048 MAC = 4096 FLOP).
// All variants produce the same four fp32 accumulators in Ampere C layout, so wall time is
// directly comparable and the ratio is the answer the gate asks for.
//
//   V0 tf32 emulation   : ema_mma_tf32 called twice  (tf32's logical op is m16n8k8, so two calls
//                         cover k=16) -> 6 mma + 2x hi/lo split + 48 shfl per unit of work
//   V1 bf16 emulation   : ema_mma_bf16 called once   -> 4 mma + bf16->f16 + 24 shfl
//   V2 single f16       : 2 volta_mma_qk (k=8 each) with pre-scaled f16 operands, fragment
//                         layout still assembled by the same 24 shfl  -> 4 mma + 24 shfl
//   V3 single f16, no shuffle : same 4 mma, operands already in Volta layout (what "direct
//                         Volta-layout loads" would buy)             -> 4 mma, ceiling
//
// V1 is the bf16 path's cost and V0 the tf32 path's cost, per the S1 layer's actual code.
// Gate: V3/V1 and V2/V1 say what removing shuffles buys on bf16 blocks; V0/V2 says what
// collapsing the 3-product tf32 split to a single f16 product buys on state_passing.
//
// fp16 prescaling (2^a, exact) is applied to the operands before the loop, so it costs nothing
// inside the measured region -- that is the whole point of the C3 prescale finding.
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <cstdio>
#include <cstdint>
#include <vector>

#include "ops/common/volta_mma.cuh"
#include "gdn_volta_mma_ref.cuh"

using ninfer::ops::volta_mma_qk;
using ninfer::ops::ema_mma_bf16;
using ninfer::ops::ema_mma_tf32;

// Warp-uniform pseudo-random operands, generated once per thread from lane + salt.
__device__ __forceinline__ unsigned mix(unsigned x) {
    x ^= x >> 16; x *= 0x7feb352du; x ^= x >> 15; x *= 0x846ca68bu; x ^= x >> 16;
    return x;
}

template <int VARIANT, int ACCS>
__global__ __launch_bounds__(256) void bench_kernel(unsigned seed, int iters, float* out) {
    const int lane = static_cast<int>(threadIdx.x) & 31;
    const int warp = (static_cast<int>(threadIdx.x) >> 5);

    // bf16 fragment regs (2 packed bf16 per u32): a[4], b[2]
    unsigned a[4], b[2];
#pragma unroll
    for (int i = 0; i < 4; ++i) { a[i] = mix(seed + lane * 0x9e3779b9u + i * 7919u); }
#pragma unroll
    for (int i = 0; i < 2; ++i) { b[i] = mix(seed + lane * 0x85ebca6bu + i * 6151u); }

    // f16 views (exact bf16 -> f16 for in-range data, as the S1 layer relies on) in Volta layout
    half2 qv[4], kv[4];
#pragma unroll
    for (int l = 0; l < 4; ++l) {
        qv[l] = __halves2half2(__float2half_rn(0.5f + 0.001f * (lane + l)),
                               __float2half_rn(0.25f + 0.002f * (lane - l)));
        kv[l] = __halves2half2(__float2half_rn(0.75f - 0.001f * (lane * 3 + l)),
                               __float2half_rn(0.125f + 0.003f * (lane + 2 * l)));
    }
    // tf32-path inputs are fp32 fragments in Ampere m16n8k8 layout
    float fa[4], fb[2];
#pragma unroll
    for (int i = 0; i < 4; ++i) { fa[i] = 0.01f * static_cast<float>((lane * 37 + i * 11) % 97) - 0.5f; }
#pragma unroll
    for (int i = 0; i < 2; ++i) { fb[i] = 0.01f * static_cast<float>((lane * 53 + i * 17) % 89) - 0.45f; }

    float c[ACCS][4];
#pragma unroll
    for (int i = 0; i < ACCS; ++i) {
#pragma unroll
        for (int j = 0; j < 4; ++j) { c[i][j] = 0.f; }
    }

    for (int it = 0; it < iters; ++it) {
#pragma unroll
        for (int acc = 0; acc < ACCS; ++acc) {
            if (VARIANT == 0) {
                // two m16n8k8 tf32 calls == one m16n8k16 unit of work
                ema_mma_tf32(c[acc][0], c[acc][1], c[acc][2], c[acc][3], fa[0], fa[1], fa[2], fa[3],
                             fb[0], fb[1]);
                ema_mma_tf32(c[acc][0], c[acc][1], c[acc][2], c[acc][3], fa[1], fa[2], fa[3], fa[0],
                             fb[1], fb[0]);
            } else if (VARIANT == 1) {
                ema_mma_bf16(c[acc][0], c[acc][1], c[acc][2], c[acc][3], a[0], a[1], a[2], a[3],
                             b[0], b[1]);
            } else if (VARIANT == 2) {
                // Single f16 product, but the Ampere->Volta fragment redistribution shuffles are
                // still there: exactly ema_mma_bf16's shuffle structure with the hi/lo machinery
                // removed. Isolates "what the 24 shuffles cost".
                using namespace ninfer::ops::gdn_ema_detail;
                unsigned a16[4] = {bf16x2_to_f16x2(a[0]), bf16x2_to_f16x2(a[1]),
                                   bf16x2_to_f16x2(a[2]), bf16x2_to_f16x2(a[3])};
                unsigned b16[2] = {bf16x2_to_f16x2(b[0]), bf16x2_to_f16x2(b[1])};
                half2 q_lo[4], q_hi[4], k_lo[4], k_hi[4];
#pragma unroll
                for (int l = 0; l < 4; ++l) {
                    const int src = (lane & 7) * 4 + l;
                    const unsigned s0 = __shfl_sync(0xFFFFFFFFu, a16[0], src);
                    const unsigned s1 = __shfl_sync(0xFFFFFFFFu, a16[1], src);
                    const unsigned s2 = __shfl_sync(0xFFFFFFFFu, a16[2], src);
                    const unsigned s3 = __shfl_sync(0xFFFFFFFFu, a16[3], src);
                    if (lane < 16) {
                        u32_ref(q_lo[l]) = (lane < 8) ? s0 : s1;
                        u32_ref(q_hi[l]) = (lane < 8) ? s2 : s3;
                    } else {
                        q_lo[l] = zero_half2();
                        q_hi[l] = zero_half2();
                    }
                }
                {
                    const int n = ((lane / 16) * 4) + (lane % 4);
#pragma unroll
                    for (int l = 0; l < 4; ++l) {
                        const int src = n * 4 + l;
                        u32_ref(k_lo[l]) = __shfl_sync(0xFFFFFFFFu, b16[0], src);
                        u32_ref(k_hi[l]) = __shfl_sync(0xFFFFFFFFu, b16[1], src);
                    }
                }
                float d0[8] = {0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f};
                volta_mma_qk(d0, q_lo, k_lo);
                volta_mma_qk(d0, q_hi, k_hi);
                c[acc][0] += d0[0];
                c[acc][1] += d0[1];
                c[acc][2] += d0[2];
                c[acc][3] += d0[3];
            } else {
                // ceiling: fragments already in Volta layout, no redistribution
                float d0[8] = {0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f};
                float d1[8] = {0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f};
                volta_mma_qk(d0, qv, kv);
                volta_mma_qk(d1, qv, kv);
                c[acc][0] += d0[4] + d1[4];
                c[acc][1] += d0[5] + d1[5];
                c[acc][2] += d0[6] + d1[6];
                c[acc][3] += d0[7] + d1[7];
            }
        }
    }

    float s = 0.f;
#pragma unroll
    for (int i = 0; i < ACCS; ++i) { s += c[i][0] + c[i][1] + c[i][2] + c[i][3]; }
    if (s == 12345.678f) { out[warp] = s; }   // never true; keeps the loop alive
}

static const char* kNames[4] = {"V0 tf32 emulation (3 products)", "V1 bf16 emulation (1 product)",
                                "V2 single f16 (shuffle-assembled)", "V3 single f16 (no shuffle)"};

template <int VARIANT>
static double run(int blocks, int iters, int accs, double* tflops_out) {
    const int threads = 256;
    float* d_out = nullptr;
    cudaMalloc(&d_out, sizeof(float) * 8);
    dim3 grid(blocks);
    bench_kernel<VARIANT, 4><<<grid, threads>>>(0x1234567u, iters, d_out);
    cudaDeviceSynchronize();
    cudaEvent_t e0, e1;
    cudaEventCreate(&e0);
    cudaEventCreate(&e1);
    cudaEventRecord(e0);
    bench_kernel<VARIANT, 4><<<grid, threads>>>(0x1234567u, iters, d_out);
    cudaEventRecord(e1);
    cudaEventSynchronize(e1);
    float ms = 0.f;
    cudaEventElapsedTime(&ms, e0, e1);
    const double warps    = double(blocks) * (threads / 32.0);
    const double macs     = warps * double(iters) * 4.0 /*ACCS*/ * 2048.0;
    const double seconds  = ms * 1e-3;
    *tflops_out           = 2.0 * macs / seconds / 1e12;
    cudaFree(d_out);
    return ms;
}

int main(int argc, char** argv) {
    int blocks = (argc > 1) ? std::atoi(argv[1]) : 320;   // 2*SM(80) * 2 blocks
    int iters  = (argc > 2) ? std::atoi(argv[2]) : 20000;

    int dev = 0;
    cudaDeviceProp prop{};
    cudaGetDeviceProperties(&prop, dev);
    printf("== S5-0 GDN mma gate · %s (sm_%d%d, %d SMs, %.0f MHz) ==\n", prop.name, prop.major,
           prop.minor, prop.multiProcessorCount, prop.clockRate / 1000.0);
    printf("   unit of work = one m16n8k16 tile-MAC (4096 FLOP); %d blocks x 256 thr, %d iters, 4 accum sets\n\n",
           blocks, iters);

    double tf[4];
    double ms[4];
    ms[0] = run<0>(blocks, iters, 4, &tf[0]);
    ms[1] = run<1>(blocks, iters, 4, &tf[1]);
    ms[2] = run<2>(blocks, iters, 4, &tf[2]);
    ms[3] = run<3>(blocks, iters, 4, &tf[3]);

    for (int i = 0; i < 4; ++i) {
        printf("  %-32s %8.2f ms   %7.2f TFLOPS\n", kNames[i], ms[i], tf[i]);
    }
    printf("\n-- gate --\n");
    printf("  V1/V0 (bf16 vs tf32 emulation) = %.2fx\n", ms[0] / ms[1]);
    printf("  V2/V1 (drop shuffles on bf16)  = %.2fx\n", ms[1] / ms[2]);
    printf("  V3/V1 (ceiling vs bf16)        = %.2fx\n", ms[1] / ms[3]);
    printf("  V0/V2 (tf32 -> single f16)     = %.2fx\n", ms[0] / ms[2]);
    printf("  V0/V3 (tf32 -> ceiling)        = %.2fx\n", ms[0] / ms[3]);
    printf("\n  fp16 peak for reference: 125 TFLOPS (V100 spec, 16x16x16 macro-MMA)\n");
    printf("GDN_MMA_GATE_DONE\n");
    return 0;
}
