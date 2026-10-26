#pragma once

// GDN chunked Volta (sm_70) emulation layer for the sm_80-only mma helpers in
// ops/common/mma.cuh. Fragment-level drop-in: the ema_* functions below take and
// return Ampere mma fragments (m16n8k16 f16/bf16 and m16n8k8 tf32 layouts, per the
// PTX ISA fragment tables) and produce equivalent results on sm_70 by
// redistributing data through warp shuffles into the repo's validated Volta
// m8n8k4 primitives (volta_mma.cuh, end-to-end verified on real V100 hardware --
// see _scratch/port-log/gdn-s1-lab-notes.md and validate_volta_qk.cu).
//
// Design notes:
//  * bf16: bf16 -> fp16 conversion is exact (8-bit mantissa fits fp16's 10) as
//    long as |x| stays under fp16 max (65504); overflow becomes inf. GDN
//    activation-side magnitudes are well inside; the B-class end-to-end oracle
//    (queue-v2 ruling #1) is the arbiter for whatever drifts.
//  * tf32: fp32 inputs are split into fp16 hi + lo parts and multiplied as
//    hh + hl + lh (ll dropped, ~2^-22 of the product). Effective precision
//    ~21 mantissa bits -- strictly higher than hardware tf32 (10), which is the
//    safe direction for the W upper-triangular inversion. Same fp16 range caveat;
//    inputs below ~6e-8 lose their lo part (GDN operands never live there).
//  * ldmatrix is NOT emulated here. The chunked kernels' smem loads are handled
//    in S2 (either per-lane LDS gather producing Ampere fragments, or direct
//    Volta-layout loads that skip the A/B redistribution shuffles entirely).
//
// Accumulation semantics match the emulated instructions: c0..c3 are read-modify-
// write accumulators in the Ampere C layout.

#include "ops/common/volta_mma.cuh"

#include <cuda_fp16.h>

namespace ninfer::ops {

#if !defined(__CUDA_ARCH__) || __CUDA_ARCH__ == 700

namespace gdn_ema_detail {

// packed bf16x2 (u32, low 16 bits = first element) -> packed f16x2 (same order).
__device__ __forceinline__ unsigned bf16x2_to_f16x2(unsigned v) {
    const float lo = __uint_as_float(v << 16);
    const float hi = __uint_as_float(v & 0xFFFF0000u);
    const __half2 h = __floats2half2_rn(lo, hi);
    return *reinterpret_cast<const unsigned*>(&h);
}

__device__ __forceinline__ unsigned u32_of(const half2& h) {
    return *reinterpret_cast<const unsigned*>(&h);
}

__device__ __forceinline__ unsigned& u32_ref(half2& h) { return *reinterpret_cast<unsigned*>(&h); }

__device__ __forceinline__ half2 zero_half2() { return __float2half2_rn(0.0f); }

// Volta D fragment (32x8 f32 across the warp; rows 0..15 live here) -> Ampere C
// layout (c0=(g,2t) c1=(g,2t+1) c2=(g+8,2t) c3=(g+8,2t+1)), accumulated into c*.
// Inverse of the validated volta_d_get_i/j scatter: D[i][j] lives at Volta lane
// (i&1)|(j&2)|(i&4) (+8 for octet 1) in register (i&2)|(j&1)|(j&4).
__device__ __forceinline__ void d_to_c_accum(const float (&d)[8], float& c0, float& c1,
                                             float& c2, float& c3) {
    const int lane = static_cast<int>(threadIdx.x) & 31;
    const int g  = lane >> 2;
    const int t  = lane & 3;
    const int vl = (g & 1) | ((t & 1) << 1) | (g & 4);  // Volta lane holding D[g][2t],D[g][2t+1]
    const int l0 = (g & 2) | ((t & 2) << 1);            // its d-register (j&4 = (t&2)<<1 for j=2t)
    float y[8];
    float z[8];
#pragma unroll
    for (int r = 0; r < 8; ++r) {
        y[r] = __shfl_sync(0xFFFFFFFFu, d[r], vl);
        z[r] = __shfl_sync(0xFFFFFFFFu, d[r], vl + 8);
    }
    const float d00 = (l0 == 0) ? y[0] : (l0 == 2) ? y[2] : (l0 == 4) ? y[4] : y[6];
    const float d01 = (l0 == 0) ? y[1] : (l0 == 2) ? y[3] : (l0 == 4) ? y[5] : y[7];
    const float d10 = (l0 == 0) ? z[0] : (l0 == 2) ? z[2] : (l0 == 4) ? z[4] : z[6];
    const float d11 = (l0 == 0) ? z[1] : (l0 == 2) ? z[3] : (l0 == 4) ? z[5] : z[7];
    c0 += d00;
    c1 += d01;
    c2 += d10;
    c3 += d11;
}

}  // namespace gdn_ema_detail

// Emulates mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 on sm_70.
// Fragment interface identical to ops::mma_bf16 (Ampere m16n8k16):
//   a[4] packed bf16x2: a0=(g,2t..2t+1) a1=(g+8,2t..2t+1) a2=(g,2t+8..2t+9) a3=(g+8,2t+8..2t+9)
//   b[2] packed bf16x2: b0=(2t..2t+1,g) b1=(2t+8..2t+9,g)   (k x n)
//   c0..c3: (g,2t) (g,2t+1) (g+8,2t) (g+8,2t+1), accumulated.
__device__ __forceinline__ void ema_mma_bf16(float& c0, float& c1, float& c2, float& c3,
                                             unsigned a0, unsigned a1, unsigned a2, unsigned a3,
                                             unsigned b0, unsigned b1) {
    const int lane = static_cast<int>(threadIdx.x) & 31;
    using namespace gdn_ema_detail;

    unsigned a16[4] = {bf16x2_to_f16x2(a0), bf16x2_to_f16x2(a1), bf16x2_to_f16x2(a2),
                       bf16x2_to_f16x2(a3)};
    unsigned b16[2] = {bf16x2_to_f16x2(b0), bf16x2_to_f16x2(b1)};

    // A (16x16) -> Volta Q fragments (lane = row, rows 16..31 zero), two k-halves.
    // Row r, k-half h: q[l] = packed (A[r][8h+2l], A[r][8h+2l+1]) = a{0|1} (h=0) or
    // a{2|3} (h=1) of Ampere lane (r&7)*4+l -- the pair sits in one u32 already.
    half2 q_lo[4];
    half2 q_hi[4];
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

    // B (16x8, k x n) -> Volta K fragments: n = volta_k_get_i(); k[l] = packed
    // (B[8h+2l][n], B[8h+2l+1][n]) = b0 (h=0) / b1 (h=1) of Ampere lane n*4+l.
    half2 k_lo[4];
    half2 k_hi[4];
    {
        const int n = ((lane / 16) * 4) + (lane % 4);
#pragma unroll
        for (int l = 0; l < 4; ++l) {
            const int src = n * 4 + l;
            u32_ref(k_lo[l]) = __shfl_sync(0xFFFFFFFFu, b16[0], src);
            u32_ref(k_hi[l]) = __shfl_sync(0xFFFFFFFFu, b16[1], src);
        }
    }

    float d[8] = {0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f};
    volta_mma_qk(d, q_lo, k_lo);  // k = 0..7
    volta_mma_qk(d, q_hi, k_hi);  // k = 8..15

    d_to_c_accum(d, c0, c1, c2, c3);
}

// Emulates mma.sync.aligned.m16n8k8.row.col.f32.tf32.tf32.f32 on sm_70 via the
// fp16 hi/lo split described in the file header.
// Fragment interface identical to ops::mma_tf32 (Ampere m16n8k8):
//   a[4] floats: a0=(g,t) a1=(g+8,t) a2=(g,t+4) a3=(g+8,t+4)      (m16 x k8)
//   b[2] floats: b0=(t,g) b1=(t+4,g)                              (k8 x n8)
//   c0..c3: same Ampere C layout as m16n8k16, accumulated.
__device__ __forceinline__ void ema_mma_tf32(float& c0, float& c1, float& c2, float& c3, float a0,
                                             float a1, float a2, float a3, float b0, float b1) {
    const int lane = static_cast<int>(threadIdx.x) & 31;
    using namespace gdn_ema_detail;

    // Gather each row's 8 k-values into per-lane fp32 arrays.
    // A[r][0..3] from a0/a1 of lanes r*4..r*4+3; A[r][4..7] from a2/a3 of the same lanes.
    // Shuffles are unconditional across the warp (full-mask sync); rows >= 16 discard.
    float arow[8] = {0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f};
#pragma unroll
    for (int c = 0; c < 4; ++c) {
        const int src = (lane & 7) * 4 + c;
        const float s0 = __shfl_sync(0xFFFFFFFFu, a0, src);
        const float s1 = __shfl_sync(0xFFFFFFFFu, a1, src);
        const float s2 = __shfl_sync(0xFFFFFFFFu, a2, src);
        const float s3 = __shfl_sync(0xFFFFFFFFu, a3, src);
        if (lane < 16) {
            arow[c]     = (lane < 8) ? s0 : s1;
            arow[c + 4] = (lane < 8) ? s2 : s3;
        }
    }
    // B[k][n]: b0 of lane n*4+k (k<4), b1 of lane n*4+(k-4) (k>=4).
    float brow[8];
    {
        const int n = ((lane / 16) * 4) + (lane % 4);
#pragma unroll
        for (int k = 0; k < 4; ++k) {
            const int src = n * 4 + k;
            brow[k]     = __shfl_sync(0xFFFFFFFFu, b0, src);
            brow[k + 4] = __shfl_sync(0xFFFFFFFFu, b1, src);
        }
    }

    // hi/lo split into fp16 fragments; three f16 matmuls: hh + hl + lh.
    half2 q_hh[4];
    half2 q_hl[4];
    half2 k_hh[4];
    half2 k_hl[4];
    if (lane < 16) {
#pragma unroll
        for (int l = 0; l < 4; ++l) {
            const float x0 = arow[2 * l];
            const float x1 = arow[2 * l + 1];
            const __half xh0 = __float2half_rn(x0);
            const __half xh1 = __float2half_rn(x1);
            q_hh[l] = __halves2half2(xh0, xh1);
            q_hl[l] = __floats2half2_rn(x0 - __half2float(xh0), x1 - __half2float(xh1));
        }
    } else {
#pragma unroll
        for (int l = 0; l < 4; ++l) {
            q_hh[l] = zero_half2();
            q_hl[l] = zero_half2();
        }
    }
#pragma unroll
    for (int l = 0; l < 4; ++l) {
        const float y0 = brow[2 * l];
        const float y1 = brow[2 * l + 1];
        const __half yh0 = __float2half_rn(y0);
        const __half yh1 = __float2half_rn(y1);
        k_hh[l] = __halves2half2(yh0, yh1);
        k_hl[l] = __floats2half2_rn(y0 - __half2float(yh0), y1 - __half2float(yh1));
    }

    float d[8] = {0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f};
    volta_mma_qk(d, q_hh, k_hh);
    volta_mma_qk(d, q_hh, k_hl);
    volta_mma_qk(d, q_hl, k_hh);

    d_to_c_accum(d, c0, c1, c2, c3);
}

// Bit-operand variant matching ops::mma_tf32_bits.
__device__ __forceinline__ void ema_mma_tf32_bits(float& c0, float& c1, float& c2, float& c3,
                                                  unsigned a0, unsigned a1, unsigned a2,
                                                  unsigned a3, unsigned b0, unsigned b1) {
    ema_mma_tf32(c0, c1, c2, c3, __uint_as_float(a0), __uint_as_float(a1), __uint_as_float(a2),
                 __uint_as_float(a3), __uint_as_float(b0), __uint_as_float(b1));
}

// --- ldmatrix emulation ---------------------------------------------------------------
//
// Emulates ldmatrix.sync.aligned.m8n8.xN.shared.b16{.trans} on sm_70 via per-lane
// ld.shared gathers. Address protocol identical to the real instruction: lane 8t+r
// supplies the shared-memory byte address of row r of tile t (16 contiguous bytes).
// Fragment layout per tile: non-trans lane l holds elements (r=l>>2, c=2*(l&3)..+1);
// .trans lane l holds the transposed tile's (r=2*(l&3), c=l>>2) pair, i.e. source
// elements (2*(l&3), l>>2) and (2*(l&3)+1, l>>2) from two row addresses.
//
// The emulated loads do not reproduce the hardware's bank-conflict pattern; the
// tiles these consume were laid out for conflict-free ldmatrix, so the gathers may
// take extra shared-memory cycles. Correctness is unaffected.

__device__ __forceinline__ unsigned ema_ld_shared_u32(unsigned saddr) {
    unsigned v;
    asm volatile("ld.shared.b32 %0, [%1];" : "=r"(v) : "r"(saddr));
    return v;
}

__device__ __forceinline__ unsigned short ema_ld_shared_u16(unsigned saddr) {
    unsigned short v;
    asm volatile("ld.shared.u16 %0, [%1];" : "=h"(v) : "r"(saddr));
    return v;
}

__device__ __forceinline__ void ema_ldmatrix_x2(unsigned& r0, unsigned& r1, unsigned addr) {
    const int lane = static_cast<int>(threadIdx.x) & 31;
    const int row  = lane >> 2;
    const int col  = (lane & 3) << 1;  // element base within the row (2 bytes each)
    const unsigned a0 = __shfl_sync(0xFFFFFFFFu, addr, row);
    const unsigned a1 = __shfl_sync(0xFFFFFFFFu, addr, 8 + row);
    r0 = ema_ld_shared_u32(a0 + col * 2);
    r1 = ema_ld_shared_u32(a1 + col * 2);
}

__device__ __forceinline__ void ema_ldmatrix_x4(unsigned& r0, unsigned& r1, unsigned& r2,
                                                unsigned& r3, unsigned addr) {
    const int lane = static_cast<int>(threadIdx.x) & 31;
    const int row  = lane >> 2;
    const int col  = (lane & 3) << 1;
    unsigned t[4];
#pragma unroll
    for (int tile = 0; tile < 4; ++tile) {
        const unsigned ra = __shfl_sync(0xFFFFFFFFu, addr, (tile << 3) + row);
        t[tile] = ema_ld_shared_u32(ra + col * 2);
    }
    r0 = t[0];
    r1 = t[1];
    r2 = t[2];
    r3 = t[3];
}

__device__ __forceinline__ void ema_ldmatrix_x2_t(unsigned& r0, unsigned& r1, unsigned addr) {
    const int lane = static_cast<int>(threadIdx.x) & 31;
    const int row0 = (lane & 3) << 1;  // source rows 2*(lane&3), +1
    const int col  = lane >> 2;        // source column
    const unsigned a00 = __shfl_sync(0xFFFFFFFFu, addr, row0);
    const unsigned a01 = __shfl_sync(0xFFFFFFFFu, addr, row0 + 1);
    const unsigned a10 = __shfl_sync(0xFFFFFFFFu, addr, 8 + row0);
    const unsigned a11 = __shfl_sync(0xFFFFFFFFu, addr, 8 + row0 + 1);
    const unsigned lo0 = ema_ld_shared_u16(a00 + col * 2);
    const unsigned hi0 = ema_ld_shared_u16(a01 + col * 2);
    const unsigned lo1 = ema_ld_shared_u16(a10 + col * 2);
    const unsigned hi1 = ema_ld_shared_u16(a11 + col * 2);
    r0 = lo0 | (hi0 << 16);
    r1 = lo1 | (hi1 << 16);
}

__device__ __forceinline__ void ema_ldmatrix_x4_t(unsigned& r0, unsigned& r1, unsigned& r2,
                                                  unsigned& r3, unsigned addr) {
    const int lane = static_cast<int>(threadIdx.x) & 31;
    const int row0 = (lane & 3) << 1;
    const int col  = lane >> 2;
    unsigned t[4];
#pragma unroll
    for (int tile = 0; tile < 4; ++tile) {
        const unsigned a0 = __shfl_sync(0xFFFFFFFFu, addr, (tile << 3) + row0);
        const unsigned a1 = __shfl_sync(0xFFFFFFFFu, addr, (tile << 3) + row0 + 1);
        const unsigned lo = ema_ld_shared_u16(a0 + col * 2);
        const unsigned hi = ema_ld_shared_u16(a1 + col * 2);
        t[tile] = lo | (hi << 16);
    }
    r0 = t[0];
    r1 = t[1];
    r2 = t[2];
    r3 = t[3];
}

#endif  // !defined(__CUDA_ARCH__) || __CUDA_ARCH__ == 700

}  // namespace ninfer::ops
