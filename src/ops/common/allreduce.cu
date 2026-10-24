// Implements: include/ninfer/ops/allreduce.h
//
// Host-side composition only: the transport is cudaMemcpyAsync with cudaMemcpyDeviceToDevice over
// UVA pointers (see pull_peer() below -- deliberately NOT cudaMemcpyPeerAsync, which stream
// capture rejects), and the local combine reuses the qualified residual_add computation body
// (x += y in BF16 with FP32 accumulation and a single round-to-nearest-even on store), which is
// exactly this Op's local step. Sharing that private launch body keeps one implementation of the
// BF16 sum instead of a second, separately qualified copy of the same arithmetic.
//
// Both collectives share one three-phase issue order. The phases exist because a wait must not be
// issued before the record it observes: cudaStreamWaitEvent snapshots the event's current state,
// so phase B's wait on inputs_ready[1-r] would snapshot a stale (or absent) capture point if the
// peer's phase-A record had not been issued yet.
//
//   phase A, both ranks:  record(inputs_ready[r])
//   phase B, both ranks:  wait(inputs_ready[1-r]); pull peer source into own storage;
//                         record(pull_done[r])
//   phase C, both ranks:  wait(pull_done[1-r]); local combine (allreduce_sum only)
//
// THE PULL ITSELF is cudaMemcpyAsync with cudaMemcpyDeviceToDevice over UVA pointers, NOT
// cudaMemcpyPeerAsync -- see pull_peer() below for why. The choreography, the streams each call
// is issued on, and the ordering proof are unchanged by that choice: it is the same transfer
// expressed through the API that CUDA graph capture accepts.
#include "ninfer/ops/allreduce.h"

#include "ops/common/bf16_vector.cuh"
#include "ops/common/memory.cuh"

#include <cuda_bf16.h>
#include "ops/launcher/residual_add.h" // detail::residual_add_launch

#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <fstream>
#include <stdexcept>
#include <string>
#include <utility>

namespace ninfer::ops {
namespace {

void require(bool condition, const char* message) {
    if (!condition) { throw std::invalid_argument(message); }
}

void require_two_devices(const ExecutionContext& ec, const char* message) {
    require(ec.tp == 2 && ec.dev[0].has_value() && ec.dev[1].has_value(), message);
    require(ec.dev[0]->device != ec.dev[1]->device, message);
}

std::uint8_t* byte_offset(void* base, std::size_t offset) {
    return static_cast<std::uint8_t*>(base) + offset;
}

// The inbound half of a pull: `bytes` from `source` (resident on the peer device) into
// `destination` (resident on the device `stream` belongs to), issued on the DESTINATION's stream.
//
// Deliberately NOT cudaMemcpyPeerAsync. That entry point is rejected inside a stream capture
// region with cudaErrorStreamCaptureUnsupported (measured on CUDA 13.1 / driver 580.178.04, Task
// 4.2's capture probe), which would make the entire tensor-parallel decode program uncapturable
// and cost the ~40-per-layer host launch overhead that CUDA Graphs exist to remove. Under unified
// virtual addressing -- which every 64-bit Linux CUDA context has -- a device pointer already
// names its device, so cudaMemcpyAsync with cudaMemcpyDeviceToDevice expresses exactly the same
// cross-device transfer: direct over PCIe when the driver granted peer access, transparently
// staged through host memory when it did not (GeForce-class boards), identical either way in
// bytes moved and stream ordering. Verified equal to the peer form both eagerly (this file's
// qualification suite) and under capture.
cudaError_t pull_peer(void* destination, const void* source, std::size_t bytes,
                      cudaStream_t stream) {
    return cudaMemcpyAsync(destination, source, bytes, cudaMemcpyDeviceToDevice, stream);
}

// Current-device save/restore. Both collectives issue work for each device in turn and must not
// leave the caller's current device changed.
class CurrentDeviceGuard {
public:
    CurrentDeviceGuard() { CUDA_CHECK(cudaGetDevice(&previous_)); }

    ~CurrentDeviceGuard() {
        const cudaError_t status = cudaSetDevice(previous_);
        if (status != cudaSuccess) {
            std::fprintf(stderr, "CUDA cleanup failed during cudaSetDevice: %s: %s\n",
                         cudaGetErrorName(status), cudaGetErrorString(status));
        }
    }

    CurrentDeviceGuard(const CurrentDeviceGuard&)            = delete;
    CurrentDeviceGuard& operator=(const CurrentDeviceGuard&) = delete;

    static void set(int device) { CUDA_CHECK(cudaSetDevice(device)); }

private:
    int previous_ = 0;
};

void startup_check(cudaError_t status, const char* operation) {
    if (status != cudaSuccess) {
        throw std::runtime_error(std::string("peer transport startup: ") + operation + ": " +
                                 cudaGetErrorName(status) + ": " + cudaGetErrorString(status));
    }
}

// Linux CUDA PCIe P2P is unsupported behind a translated IOMMU domain. A small
// allocation can nevertheless pass a copy probe while other mappings silently
// lose writes, so the domain restriction takes precedence over that probe.
std::string translated_iommu_domain(const ExecutionContext& ec) {
    std::string reason;
#if defined(__linux__)
    for (int rank = 0; rank < 2; ++rank) {
        char pci_bus_id[32]{};
        startup_check(cudaDeviceGetPCIBusId(pci_bus_id, sizeof(pci_bus_id), ec.dev[rank]->device),
                      "cudaDeviceGetPCIBusId");
        // CUDA may emit uppercase hexadecimal; Linux PCI sysfs names are lowercase.
        for (char& c : pci_bus_id) {
            if (c >= 'A' && c <= 'F') { c += 'a' - 'A'; }
        }
        std::ifstream domain_file(std::string("/sys/bus/pci/devices/") + pci_bus_id +
                                  "/iommu_group/type");
        std::string domain;
        domain_file >> domain;
        if (domain == "DMA" || domain == "DMA-FQ") {
            if (!reason.empty()) { reason += "; "; }
            reason += std::string("PCI ") + pci_bus_id + " uses translated IOMMU domain " + domain;
        }
    }
#else
    (void)ec;
#endif
    return reason;
}

// Startup-only storage. The exact payload catches drivers that advertise peer access
// but silently drop DMA writes (observed with V100s behind an IOMMU). Use the same
// pull API and destination compute stream as the collectives, without GPU peer loads.
class PeerTransferProbe {
public:
    explicit PeerTransferProbe(const ExecutionContext& ec) : ec_(ec) {
        startup_check(cudaGetDevice(&previous_), "cudaGetDevice");
    }

    ~PeerTransferProbe() {
        for (int rank = 0; rank < 2; ++rank) {
            cleanup(cudaSetDevice(ec_.dev[rank]->device), "cudaSetDevice");
            if (source_[rank] != nullptr) { cleanup(cudaFree(source_[rank]), "cudaFree source"); }
            if (destination_[rank] != nullptr) {
                cleanup(cudaFree(destination_[rank]), "cudaFree destination");
            }
        }
        cleanup(cudaSetDevice(previous_), "restore device");
    }

    PeerTransferProbe(const PeerTransferProbe&) = delete;
    PeerTransferProbe& operator=(const PeerTransferProbe&) = delete;

    void set_device(int rank) const {
        startup_check(cudaSetDevice(ec_.dev[rank]->device), "cudaSetDevice");
    }

    void initialize() {
        for (int rank = 0; rank < 2; ++rank) {
            set_device(rank);
            startup_check(cudaMalloc(&source_[rank], kBytes), "cudaMalloc source");
            startup_check(cudaMalloc(&destination_[rank], kBytes), "cudaMalloc destination");
            std::array<std::uint32_t, kWords> values{};
            for (std::size_t i = 0; i < kWords; ++i) { values[i] = pattern(rank, i); }
            startup_check(cudaMemcpyAsync(source_[rank], values.data(), kBytes,
                                           cudaMemcpyHostToDevice, ec_.dev[rank]->stream),
                          "initialize source");
            startup_check(cudaStreamSynchronize(ec_.dev[rank]->stream), "retire source");
        }
    }

    // Empty means both complete copies matched their independent host patterns exactly.
    std::string qualify() {
        std::string mismatch;
        for (int rank = 0; rank < 2; ++rank) {
            set_device(rank);
            cudaStream_t stream = ec_.dev[rank]->stream;
            startup_check(cudaMemsetAsync(destination_[rank], 0xcd, kBytes, stream),
                          "clear destination");
            startup_check(pull_peer(destination_[rank], source_[1 - rank], kBytes, stream),
                          "cross-device copy");
            startup_check(cudaStreamSynchronize(stream), "retire cross-device copy");
            std::array<std::uint32_t, kWords> actual{};
            startup_check(cudaMemcpy(actual.data(), destination_[rank], kBytes,
                                      cudaMemcpyDeviceToHost), "read destination");
            for (std::size_t i = 0; i < kWords; ++i) {
                if (actual[i] != pattern(1 - rank, i)) {
                    if (mismatch.empty()) {
                        mismatch = "device " + std::to_string(ec_.dev[1 - rank]->device) +
                                   " -> " + std::to_string(ec_.dev[rank]->device) +
                                   " data mismatch at word " + std::to_string(i);
                    }
                    break;
                }
            }
        }
        return mismatch;
    }

    void disable_peer_access() const {
        for (int rank = 0; rank < 2; ++rank) {
            set_device(rank);
            const cudaError_t status = cudaDeviceDisablePeerAccess(ec_.dev[1 - rank]->device);
            if (status == cudaErrorPeerAccessNotEnabled) {
                (void)cudaGetLastError();
            } else {
                startup_check(status, "cudaDeviceDisablePeerAccess");
            }
        }
    }

private:
    static constexpr std::size_t kWords = 4096;
    static constexpr std::size_t kBytes = kWords * sizeof(std::uint32_t);

    static std::uint32_t pattern(int rank, std::size_t index) {
        return 0x4f000000U ^ (std::uint32_t(rank) << 20U) ^
               (static_cast<std::uint32_t>(index) * 65537U);
    }

    static void cleanup(cudaError_t status, const char* operation) noexcept {
        if (status != cudaSuccess) {
            std::fprintf(stderr, "CUDA cleanup failed during peer probe %s: %s: %s\n",
                         operation, cudaGetErrorName(status), cudaGetErrorString(status));
        }
    }

    const ExecutionContext& ec_;
    int previous_ = 0;
    std::array<void*, 2> source_{};
    std::array<void*, 2> destination_{};
};

#ifndef NDEBUG
// Debug-only residency and aliasing predicates. These cost a driver round trip per pointer, so
// they are compiled out of the Release build the product ships; a wrong-device or self-overlapping
// argument is a caller bug that surfaces here during development instead of as a silently wrong
// result or an opaque cudaErrorInvalidValue later.
void require_resident_on(const void* pointer, int device, const char* message) {
    cudaPointerAttributes attributes{};
    CUDA_CHECK(cudaPointerGetAttributes(&attributes, pointer));
    require(attributes.type == cudaMemoryTypeDevice && attributes.device == device, message);
}

void require_disjoint(const void* first, std::size_t first_bytes, const void* second,
                      std::size_t second_bytes, const char* message) {
    const auto* a = static_cast<const std::uint8_t*>(first);
    const auto* b = static_cast<const std::uint8_t*>(second);
    require(a + first_bytes <= b || b + second_bytes <= a, message);
}
#endif

// --- One-shot direct all-reduce (see allreduce.h, ONE-SHOT ROUTE) ---------------------------------
//
// CTA c owns elements [c*kDirectSlice, (c+1)*kDirectSlice) of every call, whatever the payload
// size, so its receive region and its sequence counter are its own. Write-after-read safety: a
// rank's CTA c can begin call s+1 only after its call s finished, which needed the peer's flag s;
// the peer publishes flag s before reading its call-s data, so it may still be reading slot s&1
// while this rank writes call s+1 -- into slot (s+1)&1. It cannot write slot s&1 again (call s+2)
// before the peer's flag s+1, which the peer publishes only after finishing call s.
constexpr int kDirectThreads   = 256;
constexpr int kDirectSlicePacks = kDirectThreads;             // one 8-element pack per thread
constexpr int kDirectMaxCtas   = static_cast<int>(kDirectAllreduceMaxElements / (8 * kDirectSlicePacks));
constexpr std::size_t kDirectSlotBytes = static_cast<std::size_t>(kDirectAllreduceMaxElements) * 2;
static_assert(kDirectMaxCtas * 8 * kDirectSlicePacks == kDirectAllreduceMaxElements);
// About ten seconds of spinning at V100 clocks: far beyond any legitimate skew between the two
// ranks, so reaching it means the peer will never arrive (a caller broke the paired-call rule).
constexpr long long kDirectSpinCycles = 15'000'000'000LL;

// The copy route's combine (residual_add_pair in ops/kernel/residual_add.cuh): x + y in FP32 with
// one BF16 round-to-nearest-even. That header defines kernels, so the two lines are restated here.
__device__ __forceinline__ __nv_bfloat162 combine_pair(__nv_bfloat162 y, __nv_bfloat162 x) {
    return __floats2bfloat162_rn(__low2float(x) + __low2float(y), __high2float(x) + __high2float(y));
}

__device__ __forceinline__ unsigned load_volatile(const unsigned* p) {
    return *static_cast<const volatile unsigned*>(p);
}

__launch_bounds__(kDirectThreads) __global__ void direct_allreduce_kernel(
    Bf16x8Pack* __restrict__ x, std::int64_t packs, const Bf16x8Pack* own_receive,
    Bf16x8Pack* peer_receive, const unsigned* own_flags, unsigned* peer_flags,
    unsigned* sequence) {
    __shared__ unsigned call;
    const int cta = static_cast<int>(blockIdx.x);
    if (threadIdx.x == 0) { call = sequence[cta] + 1U; }
    __syncthreads();
    const unsigned s    = call;
    const std::int64_t i = static_cast<std::int64_t>(cta) * kDirectSlicePacks + threadIdx.x;
    const std::int64_t slot = static_cast<std::int64_t>(s & 1U) * (kDirectAllreduceMaxElements / 8);
    const bool live = i < packs;
    Bf16x8Pack own{};
    if (live) {
        own = x[i];
        peer_receive[slot + i] = own;
    }
    __threadfence_system();
    __syncthreads();
    if (threadIdx.x == 0) {
        *static_cast<volatile unsigned*>(peer_flags + cta) = s;
        const long long started = clock64();
        while (static_cast<int>(load_volatile(own_flags + cta) - s) < 0) {
            if (clock64() - started > kDirectSpinCycles) { __trap(); }
        }
        __threadfence();
        sequence[cta] = s;
    }
    __syncthreads();
    if (live) {
        const uint4 raw = __ldcg(reinterpret_cast<const uint4*>(own_receive + slot + i));
        Bf16x8Pack peer;
        static_assert(sizeof(peer) == sizeof(raw));
        memcpy(&peer, &raw, sizeof(raw));
#pragma unroll
        for (int pair = 0; pair < 4; ++pair) {
            own.pair[pair] = combine_pair(peer.pair[pair], own.pair[pair]);
        }
        x[i] = own;
    }
}

bool direct_admits(const std::array<Tensor, 2>& buffer, std::int64_t count) {
    if (count <= 0 || count % 8 != 0 || count > kDirectAllreduceMaxElements) { return false; }
    for (const Tensor& t : buffer) {
        if ((reinterpret_cast<std::uintptr_t>(t.data) & 15U) != 0) { return false; }
    }
    return true;
}

// --- Two-epoch packet all-reduce (TP2) ------------------------------------------------------------
//
// A second direct-P2P schedule alongside the one-shot route above, aimed at the small payloads that
// dominate a TP2 decode round. It keeps the local contribution in registers, publishes only the
// remote copy into the peer's inbox, and reduces with the same FP32 sum plus one BF16 rounding, so
// the result is bit-identical to both other routes.
//
// Two differences from the one-shot route carry the win. First, the arrival signal is per 16-byte
// pack -- a reserved BF16 NaN pattern (0xffff) marks an empty lane, so every offset has exactly one
// producer and one consumer and no CTA-wide flag rendezvous is needed. Second, the inbox is split
// into two fixed-pitch epochs; a rank publishing epoch e+1 writes disjoint addresses while the peer
// is still reading epoch e, so neither rank blocks the other between consecutive calls. Completing
// epoch e requires the peer to have produced e, therefore neither rank can recycle e+2's addresses
// before the peer consumed and retired e. The pitch and the block-to-offset mapping are fixed for
// every message size, so the schedule replays unchanged inside captured graphs.
constexpr int kPacketThreads          = 64;
constexpr std::size_t kPacketMaxBytes = 81920;
constexpr int kPacketBlocks   = static_cast<int>(kPacketMaxBytes / (sizeof(Bf16x8Pack) * kPacketThreads));
constexpr int kPacketEpochPacks = static_cast<int>(kPacketMaxBytes / sizeof(Bf16x8Pack));
static_assert(kPacketBlocks * kPacketThreads * sizeof(Bf16x8Pack) == kPacketMaxBytes);
static_assert(kPacketBlocks * kPacketThreads == kPacketEpochPacks);
constexpr std::size_t kPacketBytes = 2 * kPacketMaxBytes + kPacketBlocks * sizeof(unsigned);
// Same bound as the one-shot route: far beyond any legitimate skew between the two ranks.
constexpr long long kPacketSpinCycles = 15'000'000'000LL;

// Volatile vector access: the inbox is written by the peer's kernel and read here, so neither side
// may be served from (or coalesced through) a non-coherent cache level.
__device__ __forceinline__ Bf16x8Pack allreduce_packet_load(const Bf16x8Pack* address) {
    uint4 bits;
    asm volatile("ld.volatile.global.v4.b32 {%0,%1,%2,%3},[%4];"
                 : "=r"(bits.x), "=r"(bits.y), "=r"(bits.z), "=r"(bits.w)
                 : "l"(address));
    return *reinterpret_cast<const Bf16x8Pack*>(&bits);
}

__device__ __forceinline__ void allreduce_packet_store(Bf16x8Pack* address,
                                                       const Bf16x8Pack& value) {
    const auto& bits = *reinterpret_cast<const uint4*>(&value);
    asm volatile("st.volatile.global.v4.b32 [%4],{%0,%1,%2,%3};"
                 : : "r"(bits.x), "r"(bits.y), "r"(bits.z), "r"(bits.w), "l"(address)
                 : "memory");
}

__launch_bounds__(kPacketThreads) __global__ void allreduce_packet_sum_kernel(
    Bf16x8Pack* __restrict__ buffer, Bf16x8Pack* local, Bf16x8Pack* peer, unsigned* epochs,
    int packs, int epoch_packs) {
    const int i = static_cast<int>(blockIdx.x) * kPacketThreads + static_cast<int>(threadIdx.x);
    const unsigned epoch = epochs[blockIdx.x] & 1U;
    if (i < packs) {
        Bf16x8Pack a = load_vec<Bf16x8Pack>(buffer + i);
        Bf16x8Pack empty;
        auto* a_bits     = reinterpret_cast<std::uint16_t*>(&a);
        auto* empty_bits = reinterpret_cast<std::uint16_t*>(&empty);
#pragma unroll
        for (int j = 0; j < 8; ++j) {
            // A payload lane that already holds the empty marker would be indistinguishable from an
            // unanswered one; canonicalising it to quiet NaN keeps every finite pattern unchanged.
            if (a_bits[j] == 0xffffU) { a_bits[j] = 0x7fc0U; }
            empty_bits[j] = 0xffffU;
        }
        const std::int64_t offset = static_cast<std::int64_t>(epoch) * epoch_packs + i;
        allreduce_packet_store(peer + offset, a);
        Bf16x8Pack b;
        const long long started = clock64();
        for (;;) {
            b = allreduce_packet_load(local + offset);
            const auto* bits = reinterpret_cast<const std::uint16_t*>(&b);
            bool missing     = false;
#pragma unroll
            for (int j = 0; j < 8; ++j) { missing |= bits[j] == 0xffffU; }
            if (!missing) { break; }
            if (clock64() - started > kPacketSpinCycles) { __trap(); }
        }
#pragma unroll
        for (int j = 0; j < 4; ++j) {
            a.pair[j] = combine_pair(b.pair[j], a.pair[j]);
        }
        store_vec(buffer + i, a);
        allreduce_packet_store(local + offset, empty);
    }
    __syncthreads();
    if (threadIdx.x == 0) { epochs[blockIdx.x] = epoch ^ 1U; }
}

bool packet_admits(const std::array<Tensor, 2>& buffer, std::int64_t count) {
    if (count <= 0 || count % 8 != 0) { return false; }
    if (static_cast<std::size_t>(count) > kPacketEpochPacks * 8U) { return false; }
    for (const Tensor& t : buffer) {
        if ((reinterpret_cast<std::uintptr_t>(t.data) & 15U) != 0) { return false; }
    }
    return true;
}

} // namespace

bool enable_peer_access(const ExecutionContext& ec) {
    if (ec.tp != 2 || !ec.dev[0].has_value() || !ec.dev[1].has_value()) { return false; }
    const int pair[2] = {ec.dev[0]->device, ec.dev[1]->device};
    if (pair[0] == pair[1]) { return false; }

    PeerTransferProbe probe(ec);
    try {
        std::string direct_failure = translated_iommu_domain(ec);
        int forward = 0;
        int reverse = 0;
        if (direct_failure.empty()) {
            startup_check(cudaDeviceCanAccessPeer(&forward, pair[0], pair[1]),
                          "cudaDeviceCanAccessPeer forward");
            startup_check(cudaDeviceCanAccessPeer(&reverse, pair[1], pair[0]),
                          "cudaDeviceCanAccessPeer reverse");
        }
        const bool supported = direct_failure.empty() && forward != 0 && reverse != 0;
        if (supported) {
            for (int rank = 0; rank < 2; ++rank) {
                probe.set_device(rank);
                const cudaError_t status = cudaDeviceEnablePeerAccess(pair[1 - rank], 0);
                if (status == cudaErrorPeerAccessAlreadyEnabled) {
                    (void)cudaGetLastError();
                } else {
                    startup_check(status, "cudaDeviceEnablePeerAccess");
                }
            }
        } else {
            if (direct_failure.empty()) { direct_failure = "peer access unavailable"; }
            probe.disable_peer_access();
        }
        probe.initialize();
        if (supported) {
            direct_failure = probe.qualify();
            if (direct_failure.empty()) { return true; }
            probe.disable_peer_access();
        }

        const std::string staged_failure = probe.qualify();
        if (!staged_failure.empty()) {
            throw std::runtime_error("peer transport startup: host-staged validation failed: " +
                                     staged_failure);
        }
        std::fprintf(stderr,
                     "[ninfer] direct P2P disabled (%s); using verified CUDA host-staged copies\n",
                     direct_failure.c_str());
        return false;
    } catch (...) {
        // Failed startup must not leave a partially enabled pair behind. Clear the
        // runtime's last error before cleanup; a fatal context error still prevents
        // further use, and the original startup exception remains authoritative.
        (void)cudaGetLastError();
        try { probe.disable_peer_access(); } catch (...) {}
        throw;
    }
}

PeerEvents::PeerEvents(const ExecutionContext& ec, bool direct_peer) {
    require_two_devices(ec, "PeerEvents: requires an ExecutionContext with two distinct devices");
    const CurrentDeviceGuard guard;
    // Create through a local table so a mid-way failure destroys what was already created instead
    // of leaking it; only a fully constructed set is published into the members.
    cudaEvent_t created[4] = {nullptr, nullptr, nullptr, nullptr};
    for (int slot = 0; slot < 4; ++slot) {
        const int rank             = slot % 2;
        const cudaError_t creation = cudaSetDevice(ec.dev[rank]->device);
        cudaError_t status         = creation;
        if (status == cudaSuccess) {
            status = cudaEventCreateWithFlags(&created[slot], cudaEventDisableTiming);
        }
        if (status != cudaSuccess) {
            for (int done = 0; done < slot; ++done) { cudaEventDestroy(created[done]); }
            throw std::runtime_error(std::string("PeerEvents: event creation failed: ") +
                                     cudaGetErrorName(status) + ": " + cudaGetErrorString(status));
        }
    }
    inputs_ready_ = {created[0], created[1]};
    pull_done_    = {created[2], created[3]};
    devices_      = {ec.dev[0]->device, ec.dev[1]->device};
    if (!direct_peer) { return; }
    // Setup-time allocation, before any capture: two one-shot receive slots, the two-epoch packet
    // inbox with its per-block epoch counters, then the flags and sequence counters.
    const std::size_t words = 2 * static_cast<std::size_t>(kDirectMaxCtas);
    const std::size_t bytes = 2 * kDirectSlotBytes + kPacketBytes + words * sizeof(unsigned);
    try {
        for (int rank = 0; rank < 2; ++rank) {
            CurrentDeviceGuard::set(devices_[static_cast<std::size_t>(rank)]);
            void* storage = nullptr;
            CUDA_CHECK(cudaMalloc(&storage, bytes));
            direct_.receive[static_cast<std::size_t>(rank)] = storage;
            auto* packet_base = static_cast<std::uint8_t*>(storage) + 2 * kDirectSlotBytes;
            direct_.packets[static_cast<std::size_t>(rank)] = packet_base;
            direct_.packet_epochs[static_cast<std::size_t>(rank)] =
                reinterpret_cast<unsigned*>(packet_base + 2 * kPacketMaxBytes);
            auto* words_base =
                reinterpret_cast<unsigned*>(packet_base + kPacketBytes);
            direct_.flags[static_cast<std::size_t>(rank)]    = words_base;
            direct_.sequence[static_cast<std::size_t>(rank)] = words_base + kDirectMaxCtas;
            CUDA_CHECK(cudaMemset(storage, 0, bytes));
            // An unanswered packet lane is an all-ones BF16 pattern, so the inbox starts empty
            // rather than zeroed. Everything else (slots, counters, flags) starts at zero.
            CUDA_CHECK(cudaMemset(packet_base, 0xff, 2 * kPacketMaxBytes));
            // The compute streams are non-blocking; retire the legacy-stream fill before use.
            CUDA_CHECK(cudaDeviceSynchronize());
        }
    } catch (...) {
        release_direct();
        for (cudaEvent_t event : created) { (void)cudaEventDestroy(event); }
        inputs_ready_ = {nullptr, nullptr};
        pull_done_    = {nullptr, nullptr};
        throw;
    }
}

void PeerEvents::release_direct() noexcept {
    for (int rank = 0; rank < 2; ++rank) {
        void*& storage = direct_.receive[static_cast<std::size_t>(rank)];
        if (storage == nullptr) { continue; }
        int previous = 0;
        (void)cudaGetDevice(&previous);
        (void)cudaSetDevice(devices_[static_cast<std::size_t>(rank)]);
        const cudaError_t status = cudaFree(storage);
        if (status != cudaSuccess) {
            std::fprintf(stderr, "CUDA cleanup failed during cudaFree: %s: %s\n",
                         cudaGetErrorName(status), cudaGetErrorString(status));
        }
        (void)cudaSetDevice(previous);
        storage = nullptr;
    }
    direct_ = DirectChannel{};
}

PeerEvents::~PeerEvents() {
    release_direct();
    for (std::array<cudaEvent_t, 2>* group : {&inputs_ready_, &pull_done_}) {
        for (cudaEvent_t& event : *group) {
            if (event == nullptr) { continue; }
            const cudaError_t status = cudaEventDestroy(event);
            if (status != cudaSuccess) {
                std::fprintf(stderr, "CUDA cleanup failed during cudaEventDestroy: %s: %s\n",
                             cudaGetErrorName(status), cudaGetErrorString(status));
            }
            event = nullptr;
        }
    }
}

PeerEvents::PeerEvents(PeerEvents&& other) noexcept
    : inputs_ready_(other.inputs_ready_), pull_done_(other.pull_done_), devices_(other.devices_),
      direct_(other.direct_) {
    other.inputs_ready_ = {nullptr, nullptr};
    other.pull_done_    = {nullptr, nullptr};
    other.direct_       = DirectChannel{};
}

PeerEvents& PeerEvents::operator=(PeerEvents&& other) noexcept {
    // Swap rather than destroy-then-assign: `other`'s destructor releases whatever this instance
    // held, in exactly one place.
    inputs_ready_.swap(other.inputs_ready_);
    pull_done_.swap(other.pull_done_);
    std::swap(devices_, other.devices_);
    std::swap(direct_, other.direct_);
    return *this;
}

void allreduce_sum(const std::array<Tensor, 2>& buffer, const std::array<Tensor, 2>& staging,
                   const ExecutionContext& ec, const PeerEvents& events) {
    require_two_devices(ec,
                        "allreduce_sum: requires an ExecutionContext with two distinct devices");
    for (int rank = 0; rank < 2; ++rank) {
        require(buffer[rank].dtype == DType::BF16 && staging[rank].dtype == DType::BF16,
                "allreduce_sum: buffer/staging must be BF16");
        require(buffer[rank].data != nullptr && staging[rank].data != nullptr,
                "allreduce_sum: buffer/staging data must be non-null");
        require(buffer[rank].is_contiguous() && staging[rank].is_contiguous(),
                "allreduce_sum: buffer/staging must be contiguous");
        for (int d = 0; d < 4; ++d) {
            require(buffer[rank].ne[d] == buffer[0].ne[d] && staging[rank].ne[d] == buffer[0].ne[d],
                    "allreduce_sum: buffer/staging shapes must match on both devices");
        }
    }
    require(events.live(), "allreduce_sum: events must be live");

    const std::size_t bytes = buffer[0].bytes();
    if (bytes == 0) { return; }
    const std::int64_t count = buffer[0].numel();

#ifndef NDEBUG
    for (int rank = 0; rank < 2; ++rank) {
        require_resident_on(buffer[rank].data, ec.dev[rank]->device,
                            "allreduce_sum: buffer[r] must be resident on ec.dev[r]");
        require_resident_on(staging[rank].data, ec.dev[rank]->device,
                            "allreduce_sum: staging[r] must be resident on ec.dev[r]");
        require_disjoint(buffer[rank].data, bytes, staging[rank].data, bytes,
                         "allreduce_sum: staging[r] must not overlap buffer[r]");
    }
#endif

    const CurrentDeviceGuard guard;

    const PeerEvents::DirectChannel* direct = events.direct();

    // Small payloads (the decode-round collectives) take the two-epoch packet schedule; anything
    // larger keeps the one-shot CTA-slice route. Both reduce to the same bits.
    if (direct != nullptr && packet_admits(buffer, count)) {
        const int packs = static_cast<int>(count / 8);
        const unsigned blocks =
            static_cast<unsigned>((packs + kPacketThreads - 1) / kPacketThreads);
        for (int rank = 0; rank < 2; ++rank) {
            const auto r = static_cast<std::size_t>(rank);
            CurrentDeviceGuard::set(ec.dev[rank]->device);
            allreduce_packet_sum_kernel<<<blocks, kPacketThreads, 0, ec.dev[rank]->stream>>>(
                static_cast<Bf16x8Pack*>(buffer[r].data),
                static_cast<Bf16x8Pack*>(direct->packets[r]),
                static_cast<Bf16x8Pack*>(direct->packets[1 - r]), direct->packet_epochs[r],
                packs, kPacketEpochPacks);
            CUDA_CHECK(cudaGetLastError());
        }
        return;
    }

    if (direct != nullptr && direct_admits(buffer, count)) {
        const std::int64_t packs = count / 8;
        const unsigned ctas =
            static_cast<unsigned>((packs + kDirectSlicePacks - 1) / kDirectSlicePacks);
        for (int rank = 0; rank < 2; ++rank) {
            const auto r = static_cast<std::size_t>(rank);
            CurrentDeviceGuard::set(ec.dev[rank]->device);
            direct_allreduce_kernel<<<ctas, kDirectThreads, 0, ec.dev[rank]->stream>>>(
                static_cast<Bf16x8Pack*>(buffer[r].data), packs,
                static_cast<const Bf16x8Pack*>(direct->receive[r]),
                static_cast<Bf16x8Pack*>(direct->receive[1 - r]), direct->flags[r],
                direct->flags[1 - r], direct->sequence[r]);
            CUDA_CHECK(cudaGetLastError());
        }
        return;
    }

    // Phase A: publish "my operand is complete" on each stream, before any wait observes it.
    for (int rank = 0; rank < 2; ++rank) {
        const DeviceContext& local = *ec.dev[rank];
        CurrentDeviceGuard::set(local.device);
        CUDA_CHECK(cudaEventRecord(events.inputs_ready(rank), local.stream));
    }

    // Phase B: each rank pulls the peer's operand into storage only it owns. Both inbound copies
    // are issued before either rank waits again, so the two directions overlap.
    for (int rank = 0; rank < 2; ++rank) {
        const DeviceContext& local = *ec.dev[rank];
        CurrentDeviceGuard::set(local.device);
        CUDA_CHECK(cudaStreamWaitEvent(local.stream, events.inputs_ready(1 - rank), 0));
        CUDA_CHECK(pull_peer(staging[rank].data, buffer[1 - rank].data, bytes, local.stream));
        CUDA_CHECK(cudaEventRecord(events.pull_done(rank), local.stream));
    }

    // Phase C: the in-place combine may only overwrite buffer[rank] once the peer has finished
    // reading it. That same wait is what makes the next call's phase B safe.
    for (int rank = 0; rank < 2; ++rank) {
        const DeviceContext& local = *ec.dev[rank];
        CurrentDeviceGuard::set(local.device);
        CUDA_CHECK(cudaStreamWaitEvent(local.stream, events.pull_done(1 - rank), 0));
        Tensor accumulator = buffer[rank];
        detail::residual_add_launch(staging[rank], accumulator, local.stream);
    }
}

void allgather_rows(const std::array<Tensor, 2>& destination, const std::array<Tensor, 2>& part,
                    const ExecutionContext& ec, const PeerEvents& events) {
    require_two_devices(ec,
                        "allgather_rows: requires an ExecutionContext with two distinct devices");
    const DType dtype             = destination[0].dtype;
    const std::int32_t row_length = destination[0].ne[0];
    const std::int32_t total_rows = destination[0].ne[1];
    for (int rank = 0; rank < 2; ++rank) {
        require(destination[rank].dtype == dtype && part[rank].dtype == dtype,
                "allgather_rows: destination/part must share one dtype");
        require(destination[rank].data != nullptr && part[rank].data != nullptr,
                "allgather_rows: destination/part data must be non-null");
        require(destination[rank].is_contiguous() && part[rank].is_contiguous(),
                "allgather_rows: destination/part must be contiguous");
        require(destination[rank].ne[0] == row_length && part[rank].ne[0] == row_length,
                "allgather_rows: destination/part must agree on row length ne[0]");
        require(destination[rank].ne[1] == total_rows,
                "allgather_rows: both destinations must have the same row count");
        require(destination[rank].ne[2] == 1 && destination[rank].ne[3] == 1 &&
                    part[rank].ne[2] == 1 && part[rank].ne[3] == 1,
                "allgather_rows: destination/part must be two-dimensional [C, R]");
    }
    require(part[0].ne[1] + part[1].ne[1] == total_rows,
            "allgather_rows: owned row counts must sum to the destination row count");
    require(events.live(), "allgather_rows: events must be live");

    const std::size_t row_bytes = static_cast<std::size_t>(row_length) * dtype_size(dtype);
    const std::size_t block[2]  = {row_bytes * static_cast<std::size_t>(part[0].ne[1]),
                                   row_bytes * static_cast<std::size_t>(part[1].ne[1])};
    const std::size_t offset[2] = {0, block[0]};

#ifndef NDEBUG
    for (int rank = 0; rank < 2; ++rank) {
        require_resident_on(destination[rank].data, ec.dev[rank]->device,
                            "allgather_rows: destination[r] must be resident on ec.dev[r]");
        require_resident_on(part[rank].data, ec.dev[rank]->device,
                            "allgather_rows: part[r] must be resident on ec.dev[r]");
        require_disjoint(destination[rank].data, destination[rank].bytes(), part[rank].data,
                         block[rank], "allgather_rows: part[r] must not overlap destination[r]");
    }
#endif

    const CurrentDeviceGuard guard;

    // Phase A: publish "my block is complete".
    for (int rank = 0; rank < 2; ++rank) {
        const DeviceContext& local = *ec.dev[rank];
        CurrentDeviceGuard::set(local.device);
        CUDA_CHECK(cudaEventRecord(events.inputs_ready(rank), local.stream));
    }

    // Phase B: rank r writes its own block locally and pulls the peer's block, both on its own
    // stream, so destination[r] has exactly one writer.
    for (int rank = 0; rank < 2; ++rank) {
        const DeviceContext& local = *ec.dev[rank];
        CurrentDeviceGuard::set(local.device);
        CUDA_CHECK(cudaStreamWaitEvent(local.stream, events.inputs_ready(1 - rank), 0));
        CUDA_CHECK(cudaMemcpyAsync(byte_offset(destination[rank].data, offset[rank]),
                                   part[rank].data, block[rank], cudaMemcpyDeviceToDevice,
                                   local.stream));
        CUDA_CHECK(pull_peer(byte_offset(destination[rank].data, offset[1 - rank]),
                             part[1 - rank].data, block[1 - rank], local.stream));
        CUDA_CHECK(cudaEventRecord(events.pull_done(rank), local.stream));
    }

    // Phase C: the Op writes nothing else, but the caller (or the next call) will overwrite
    // part[rank]. Ordering each stream after the peer's read is what makes that safe without a
    // host synchronization.
    for (int rank = 0; rank < 2; ++rank) {
        const DeviceContext& local = *ec.dev[rank];
        CurrentDeviceGuard::set(local.device);
        CUDA_CHECK(cudaStreamWaitEvent(local.stream, events.pull_done(1 - rank), 0));
    }
}

} // namespace ninfer::ops
