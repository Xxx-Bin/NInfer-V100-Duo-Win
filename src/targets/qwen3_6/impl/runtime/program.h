#pragma once
#include "targets/qwen3_6/impl/runtime/instance.h"
// Qwen3.6 family runtime implementation; instantiated only by exact variants.

#include "core/arena.h"
#include "core/gdn_replay_records.h"
#include "ninfer/ops/allreduce.h"
#include "ninfer/ops/sampling.h"
#include "core/decode_graph.h"
#include <ninfer/targets/qwen3_6/prepared_prompt.h>

#include "targets/qwen3_6/impl/runtime/layouts.h"
#include "targets/qwen3_6/impl/runtime/dflash_context.h"
#include "targets/qwen3_6/impl/runtime/linear_state_slots.h"
#include "targets/qwen3_6/impl/runtime/prefix_identity.h"
#include "targets/qwen3_6/impl/runtime/text_context.h"
#include "targets/qwen3_6/impl/runtime/vision_context.h"
#include "targets/qwen3_6/impl/runtime/vision_prefill.h"

#include <cstdint>
#include <array>
#include <cstddef>
#include <memory>
#include <optional>
#include <span>
#include <vector>

namespace ninfer::targets::qwen3_6::detail::NINFER_QWEN36_RUNTIME_NS {

using PreparedPromptData    = qwen3_6::PreparedPromptData;
using RewriteCheckpointKind = qwen3_6::RewriteCheckpointKind;
using RewriteCheckpointSpec = qwen3_6::RewriteCheckpointSpec;

using ReusePath = ninfer::PrefixReusePath;

[[nodiscard]] constexpr bool is_rewrite_checkpoint_restore(ReusePath path) noexcept {
    return path == ReusePath::RestoreTurnCheckpoint || path == ReusePath::RestoreResponseCheckpoint;
}

[[nodiscard]] constexpr ReusePath restore_path(RewriteCheckpointKind kind) noexcept {
    return kind == RewriteCheckpointKind::TurnClosure ? ReusePath::RestoreTurnCheckpoint
                                                      : ReusePath::RestoreResponseCheckpoint;
}

enum class RewriteCheckpointAction : std::uint8_t {
    Drop,
    KeepExisting,
    ReclassifyExisting,
    CaptureNew,
    DeferCapture,
};

enum class MtpBridgeMode : std::uint8_t {
    None,
    BeforeSuffix,
    AfterExactHit,
};

} // namespace ninfer::targets::qwen3_6::detail::NINFER_QWEN36_RUNTIME_NS

namespace ninfer::targets::qwen3_6::detail {

template <>
struct RequestBasePlanImpl<NINFER_QWEN36_VARIANT> {
    runtime::RequestPlanSummary summary;
    ops::SamplingConfig sampling;
    std::uint32_t text_kv_page_entitlement    = 0;
    std::uint32_t backend_kv_page_entitlement = 0;
    std::shared_ptr<const qwen3_6::VisionControl> vision_control;
    std::size_t vision_transient_bytes = 0;
    std::optional<qwen3_6::RewriteCheckpointSpec> rewrite_checkpoint;
    bool allow_prefix_reuse = false;
};

template <>
struct RequestPlanImpl<NINFER_QWEN36_VARIANT> {
    runtime::RequestPlanSummary summary;
    NINFER_QWEN36_RUNTIME_NS::ReusePath reuse = NINFER_QWEN36_RUNTIME_NS::ReusePath::FullReset;
    std::uint32_t reuse_base                  = 0;
    NINFER_QWEN36_RUNTIME_NS::MtpBridgeMode mtp_bridge =
        NINFER_QWEN36_RUNTIME_NS::MtpBridgeMode::None;
    bool prepare_mtp = false;
    std::optional<NINFER_QWEN36_RUNTIME_NS::VisionPrefillPlan> vision;
    // Set when the selected reuse base lives in the detached prefix registry rather than in the
    // lane. The admission path must attach that entry to its lane before prefill starts; -1 means
    // the base is the lane's own retained sequence (or there is no base at all).
    std::int32_t attach_registry_index = -1;
    NINFER_QWEN36_RUNTIME_NS::RewriteCheckpointAction rewrite_checkpoint_action =
        NINFER_QWEN36_RUNTIME_NS::RewriteCheckpointAction::Drop;
    std::optional<qwen3_6::RewriteCheckpointSpec> rewrite_checkpoint_capture;
    ops::SamplingConfig sampling;
    std::uint32_t text_kv_page_entitlement    = 0;
    std::uint32_t backend_kv_page_entitlement = 0;
};

} // namespace ninfer::targets::qwen3_6::detail

namespace ninfer::targets::qwen3_6::detail::NINFER_QWEN36_RUNTIME_NS {

using RequestPlanImpl     = qwen3_6::detail::RequestPlanImpl<Variant>;
using RequestBasePlanImpl = qwen3_6::detail::RequestBasePlanImpl<Variant>;

enum class PendingKind : std::uint8_t {
    None,
    Begin,
    Ordinary,
    Speculative,
};

struct PendingCandidate {
    PendingKind kind            = PendingKind::None;
    std::uint32_t base_E        = 0;
    std::uint32_t base_S        = 0;
    std::uint32_t prompt_tokens = 0;
    std::uint32_t produced      = 0;
};

enum class Lifecycle : std::uint8_t {
    Empty,
    Prefilling,
    Active,
    Pending,
    Complete,
};

struct RewriteCheckpoint {
    bool valid                 = false;
    RewriteCheckpointKind kind = RewriteCheckpointKind::TurnClosure;
    std::uint32_t frontier     = 0;
};

struct SequenceKVBundle {
    PagedKVAllocation text;
    std::optional<PagedKVAllocation> backend;
    // Rank 1's allocation in ITS OWN text KV pool at tp == 2. The two pools have identical page
    // geometry (only the per-page byte count differs, because rank 1 holds 2 of the 4 KV heads),
    // and every pool operation below is issued on both in the same order, so the two allocations
    // hold the same page ids and publish identical block tables.
    std::optional<PagedKVAllocation> text_peer;
    // Rank 1's allocation in ITS OWN MTP (backend) KV pool at tp == 2. Same lockstep argument as
    // `text_peer`: identical page geometry, every pool operation issued on both in the same order.
    std::optional<PagedKVAllocation> backend_peer;
};

/**
 * A sequence that survived the request which produced it and is no longer owned by any lane.
 *
 * `SequenceState` cannot simply be moved out of a lane: its `tail_hidden` and
 * `rewrite_checkpoint_hidden` are per-lane slices of a shared arena, and its linear-attention state
 * is addressed by lane index. A detached entry therefore owns its own storage for all three:
 *
 *   - the KV bundle moves (move-only allocation, no bytes copied, no refcount),
 *   - the two linear-attention states are copied into this entry's dedicated registry slots,
 *   - the two hidden vectors are copied out to host bytes (10 KiB each) and copied back on attach.
 *
 * A `PagedKVAllocation` is move-only and releasing it returns its pages, so exclusive ownership of
 * the bundle IS the whole liveness argument: there is no window in which the registry and a lane
 * both own the same pages.
 */
struct DetachedSequence {
    SequenceKVBundle kv;
    std::vector<TokenId> ledger;
    qwen3_6::detail::ResidentPrefixIdentity prefix_identity;
    RewriteCheckpoint rewrite_checkpoint;

    std::uint32_t execution_frontier      = 0;
    std::uint32_t text_kv_valid           = 0;
    std::uint32_t mtp_kv_valid            = 0;
    std::uint32_t dflash_context_frontier = 0;
    std::array<TokenId, qwen3_6::kMtpDecodeMaximumDrafts> mtp_drafts{};
    std::uint32_t mtp_draft_count = 0;
    std::int32_t rope_delta       = 0;
    bool tail_hidden_valid        = false;

    std::vector<std::byte> tail_hidden;
    std::vector<std::byte> rewrite_checkpoint_hidden;

    // Age counter used only to order eviction among equally sized entries.
    std::uint64_t detached_epoch = 0;
};

struct DecodeGraphProfile {
    std::uint32_t batch_size             = 1;
    std::uint32_t min_execution_frontier = 0;
    std::uint32_t max_execution_frontier = 0;
    std::uint32_t topology_class         = 0;
    DecodeGraphDefinition definition;
};

struct DecodeGraphTopology {
    std::uint32_t topology_class = 0;
    DecodeGraphExecutable executable;
    std::optional<std::size_t> installed_profile;
};

struct DecodeGraphFamily {
    std::vector<DecodeGraphProfile> profiles;
    std::vector<DecodeGraphTopology> topologies;
};

// Target model continuation for one logical sequence. This state remains meaningful after the
// request which produced it has finished, so it is deliberately separate from request lifecycle,
// output, sampling, and round-control state.
struct SequenceState {
    std::optional<SequenceKVBundle> kv;
    Tensor tail_hidden;
    Tensor rewrite_checkpoint_hidden;
    std::uint32_t lane = 0;

    std::uint32_t execution_frontier = 0;
    std::uint32_t ledger_frontier    = 0;
    std::vector<TokenId> ledger;
    qwen3_6::detail::ResidentPrefixIdentity prefix_identity;
    std::int32_t rope_delta               = 0;
    std::uint32_t text_kv_valid           = 0;
    std::uint32_t mtp_kv_valid            = 0;
    std::uint32_t dflash_context_frontier = 0;
    std::array<TokenId, qwen3_6::kMtpDecodeMaximumDrafts> mtp_drafts{};
    std::uint32_t mtp_draft_count = 0;
    bool tail_hidden_valid        = false;
    bool retained                 = false;
    RewriteCheckpoint rewrite_checkpoint;
};

// Request/round control is not retained with a reusable SequenceState. A later concurrent Engine
// gives every occupied request slot its own instance of this state.
struct RequestControl {
    Lifecycle lifecycle = Lifecycle::Empty;
    PendingCandidate pending;
    ops::SamplingConfig sampling_host;
    GenerationTimings timings;
    SpeculativeStats speculative_stats;

    struct Prefill {
        PreparedPromptData prompt;
        std::optional<VisionPrefillPlan> vision_plan;
        std::unique_ptr<schedule::VisionPrefillSession> vision;
        runtime::TransientRegion transient;
        std::optional<RewriteCheckpointSpec> rewrite_checkpoint_capture;
        std::uint32_t base               = 0;
        std::uint32_t cursor             = 0;
        std::uint32_t prompt_tokens      = 0;
        std::uint32_t initial_mtp_extent = 0;
        double elapsed_seconds           = 0.0;
        bool prepare_mtp                 = false;
        ReusePath reuse                  = ReusePath::FullReset;
        MtpBridgeMode mtp_bridge         = MtpBridgeMode::None;
    };

    std::optional<Prefill> prefill;
};

// Rank 1's complete runtime mirror: its own arenas, its own shard of the weights, its own halved
// decoder state, and its own RoundState. It owns no bookkeeping -- lanes, page accounting,
// sampling and the pinned host round buffers all live once, on rank 0.
struct PeerRuntime {
    PeerRuntime(DeviceContext& peer_device, const LoadedModelData& peer_model,
                const SequencePlanImpl& plan);

    PeerRuntime(const PeerRuntime&)            = delete;
    PeerRuntime& operator=(const PeerRuntime&) = delete;

    DeviceContext& device;
    const LoadedModelData& model;
    DeviceArena persistent;
    DeviceArena workspace_storage;
    WorkspaceArena work;
    std::unique_ptr<qwen3_6::DecoderState> decoder;
    // Rank 1's own GDN replay records: the speculative verify round records this device's own
    // head/channel shard and folds it here, so the two devices commit the same accepted prefix
    // from records neither ever exchanges.
    std::optional<GdnReplayRecords> replay_records;
    qwen3_6::RoundState io;
    std::optional<typename qwen3_6::detail::TP2DFlashExtension<Variant>::State> tp2_dflash;
    Tensor prefill_hidden;
    // Rank 1's OWN penalty counters. `ops::SamplingConfig::token_counts` is a raw device pointer,
    // and the MTP round's acceptance runs on both devices (see the replicated-accept note in
    // mtp_impl.h), so rank 1 must never be handed rank 0's. It is not a cache: rank 0 remains the
    // source of truth (ProgramImplCore::install_sampling zeroes both, and the one increment that
    // happens on rank 0 alone -- prefill's bonus token -- is copied across before the first decode
    // round), and thereafter both are advanced by the same Op over bit-identical inputs.
    Tensor token_counts;
};

class ProgramImplCore {
public:
    ProgramImplCore(const LoadedModelData& model, const LoadedModelData* peer_model,
                    const SequencePlanImpl& plan, ExecutionContext& execution);
    ~ProgramImplCore() noexcept;

    [[nodiscard]] RequestBasePlan
    plan_request_base(const PreparedPromptData& prompt,
                      const runtime::ResolvedExecutionOptions& options);
    [[nodiscard]] RequestPlan plan_request_for_lane(std::uint32_t lane,
                                                    const PreparedPromptData& prompt,
                                                    const RequestBasePlan& base);
    // The one admission decision: whether `plan` can take `lane`, and which cached bundles must be
    // released first for it to fit. Page accounting exists here and nowhere else -- callers test
    // this plan and then execute it, never re-derive feasibility from a second rule.
    [[nodiscard]] runtime::AdmissionPlan plan_lane_admission(std::uint32_t lane,
                                                             const RequestPlan& plan) const;
    // Executes `plan.reclaim` in order. Never re-tests and never allocates: the list was computed
    // against the state this call runs in.
    void apply_lane_admission(const runtime::AdmissionPlan& plan) noexcept;
    [[nodiscard]] runtime::AdmissionResources admission_capacity() const noexcept;
    [[nodiscard]] runtime::PrefillStepResult start_prefill_lane(std::uint32_t lane,
                                                                PreparedPromptData&& prompt,
                                                                RequestPlan&& plan,
                                                                runtime::TransientRegion transient);
    [[nodiscard]] runtime::PrefillStepResult advance_prefill_lane(std::uint32_t lane);
    [[nodiscard]] runtime::BatchedGeneratedRound
    decode_batch(std::span<const std::uint32_t> lanes,
                 std::span<const runtime::RoundBudget> budgets);
    void resolve_prefill_lane(std::uint32_t lane, bool terminal);
    void resolve_pending_batch(std::span<const std::uint32_t> lanes,
                               std::span<const std::uint32_t> accepted_tokens,
                               std::span<const std::uint8_t> terminal,
                               std::span<const std::uint8_t> cancelled);
    void abort_lane(std::uint32_t lane) noexcept;
    // Count of detached entries currently held. Zero whenever --prefix-registry-slots is zero.
    [[nodiscard]] std::uint32_t detached_prefix_count() const noexcept;
    [[nodiscard]] std::uint32_t detached_prefix_capacity() const noexcept {
        return registry_capacity;
    }
    [[nodiscard]] GenerationTimings generation_timings_lane(std::uint32_t lane) const noexcept;
    [[nodiscard]] SpeculativeStats speculative_stats_lane(std::uint32_t lane) const noexcept;

    [[nodiscard]] MemorySummary memory_summary() const noexcept;

    void reset_memory_peaks() noexcept;

    const LoadedModelData& model;
    ExecutionContext& execution;
    DeviceContext& device;
    const int tp;
    const std::uint32_t capacity;
    const std::uint32_t kv_capacity;
    const std::uint32_t max_concurrency;
    // Detached prefix-registry entry count. Fixed at construction: every entry owns two
    // linear-attention slots, so the pool is sized once and never resized.
    const std::uint32_t registry_capacity;
    const std::uint32_t prefill_chunk;
    const std::uint32_t draft_window;
    const SpeculativeBackend speculative_backend;
    const DType kv_dtype;
    const std::int32_t kv_quant_group;
    const ProposalHead proposal_head;
    const bool vision_enabled;
    const bool use_cuda_graph;
    const std::size_t kv_payload_bytes;
    const std::size_t gdn_state_bytes;
    const std::size_t graph_allowance_bytes;
    // Measured graph residency per device (index = rank). At tp1 only [0] is populated. At tp2 the
    // single cross-device graph materializes driver state on BOTH devices, and each is checked
    // against the SAME per-device allowance -- graph_allowance_bytes is a per-device budget, like
    // every other field in device_reservation_bytes.
    std::array<std::size_t, 2> graph_observed_bytes{0, 0};
    // Node count of ONE captured decode graph (the first profile of the captured family). At tp2
    // one graph holds both devices' nodes, so this is the direct measurement of whether the peer's
    // half of the schedule was captured rather than left out.
    std::size_t graph_node_count = 0;
    const WorkspacePlan workspace_plan;

    DeviceArena persistent;
    DeviceArena workspace_storage;
    WorkspaceArena work;
    // YaRN rotary state, resolved once at construction.
    //
    // `rope_frequency_storage[rank]` is a dedicated 32-float device allocation on rank `rank`'s OWN
    // device, made once here and never touched again: CUDA Graph capture bakes the pointer into the
    // replayed rope launch node, so it must outlive every replay, and at tp 2 each rank ropes its
    // own head-local q/k on its own device, where a pointer into the other rank's allocation is not
    // addressable. It is deliberately NOT part of the planned persistent arena: under
    // `RopeMode::Native` nothing is allocated at all, which is what keeps a native plan and its
    // memory summary byte-identical to the pre-YaRN engine.
    //
    // `rope_frequency[rank]` is the descriptor every text rope call site reads (through
    // `ExecutionCore::rope_frequency` -> `TextContext`); a null `inv_frequency` IS the native path.
    std::array<DeviceBuffer, 2> rope_frequency_storage;
    std::array<ops::RopeFrequencyOverride, 2> rope_frequency{};
    const RopeMode rope_mode;
    const std::uint32_t effective_max_context;
    const double yarn_mscale;
    std::optional<PeerRuntime> peer;
    std::optional<ops::PeerEvents> peer_events;
    // Created once at tp2 when graphs are on; forks rank 1's stream into rank 0's capture.
    std::optional<DecodeGraphPeerBridge> graph_bridge;
    std::optional<schedule::TpPeerCore> peer_core;
    std::unique_ptr<qwen3_6::DecoderState> decoder;
    std::optional<GdnReplayRecords> replay_records;
    std::optional<DFlashPersistentState> dflash;
    std::optional<typename qwen3_6::detail::TP2DFlashExtension<Variant>::State> tp2_dflash;
    qwen3_6::RoundState io;
    Tensor prefill_hidden;
    Tensor sampling_config;
    Tensor token_counts;
    Tensor tail_hidden_store;
    Tensor rewrite_checkpoint_hidden_store;

    std::array<SequenceState, kMaximumConcurrency> sequences;
    std::array<RequestControl, kMaximumConcurrency> requests;
    // Detached prefix registry. Declared after `decoder` on purpose: entries hold KV allocations
    // from the decoder's pools, so they must be destroyed first.
    std::vector<std::optional<DetachedSequence>> registry;
    std::uint64_t registry_epoch = 0;
    // Entry that must survive an eviction triggered by a displace/detach issued from inside
    // attach. -1 means nothing is protected.
    std::int32_t registry_protected_index_ = -1;

    DecodeGraphFamily ordinary_graphs;
    DecodeGraphFamily mtp_graphs;
    DecodeGraphFamily dflash_graphs;
    DecodeGraphFamily dflash2_graphs;

    PinnedHostBuffer round_host;
    TokenId* host_tokens = nullptr;
    // Debug-only capture of the round's full-vocabulary logits, raw BF16 bits. OFF by default and
    // completely inert until enable_logits_capture(true) allocates the host buffer: with it off,
    // copy_round_logits() is a predictable branch and nothing is allocated or copied, so no
    // production path (tp1 or tp2) is changed. With it on, the buffer is overwritten every time a
    // token is sampled from io.logits at prefill finalization, for both tp1 and tp2 (io is rank
    // 0's window; logits_tp2 gathers into it the same way the tp1 leaf writes it), so no
    // tp-specific code is needed. Prefill only -- the decode path is CUDA-graph-capturable and an
    // unconditional memcpy inside a captured region is rejected at capture time; a probe that
    // requests exactly one output token completes on prefill's own sampled token and never runs a
    // decode round, which is what the parity harness (tools/tp2/parity.cpp) uses. Valid only
    // immediately after Engine::generate()/wait() returns to the calling thread for a single-lane
    // engine -- concurrent lanes would overwrite it out of order.
    std::vector<std::uint16_t> logits_capture;
    bool logits_capture_enabled = false;
    std::optional<PinnedHostBuffer> ordinary_host;
    qwen3_6::OrdinaryDecodeIngress* ordinary_host_ingress = nullptr;
    qwen3_6::OrdinaryDecodeEgress* ordinary_host_egress   = nullptr;
    // Rank 1's own pinned ordinary ingress: rank 0's record with every row's
    // `sampling[row].token_counts` nulled. Rank 1 mirrors the ordinary round for its half of the
    // weights and never samples (the output head is vocabulary-split and sampling belongs to rank
    // 0), so the sampling configs in its frame are inert -- but they must not carry a rank-0
    // DEVICE address into rank 1's frame: dereferencing one from rank 1 is a silent cross-device
    // fault. A separate pinned buffer rather than a patched copy at issue time, for the same
    // reason as `mtp_peer_host_ingress`: the upload is inside the captured decode graph, which
    // re-reads this exact host address at every replay.
    std::optional<PinnedHostBuffer> ordinary_peer_host;
    qwen3_6::OrdinaryDecodeIngress* ordinary_peer_host_ingress = nullptr;
    bool peer_egress_check_enabled     = false;
    std::uint64_t peer_egress_rounds     = 0;
    std::uint64_t peer_egress_mismatches = 0;
    std::optional<PinnedHostBuffer> mtp_host;
    qwen3_6::MtpDecodeIngress* mtp_host_ingress = nullptr;
    qwen3_6::MtpDecodeEgress* mtp_host_egress   = nullptr;
    // Rank 1's own pinned MTP ingress: byte-for-byte rank 0's record except that each row's
    // `sampling[row].token_counts` names rank 1's counter lane. It has to be a separate pinned
    // buffer rather than a patched copy made at issue time, because the ingress upload is inside
    // the captured decode graph and the graph re-reads this exact host address at every replay.
    std::optional<PinnedHostBuffer> mtp_peer_host;
    qwen3_6::MtpDecodeIngress* mtp_peer_host_ingress = nullptr;
    std::optional<PinnedHostBuffer> dflash_host;
    qwen3_6::DFlashDecodeIngress* dflash_host_ingress = nullptr;
    qwen3_6::DFlashDecodeEgress* dflash_host_egress   = nullptr;
    std::optional<PinnedHostBuffer> dflash2_host;
    qwen3_6::DFlash2DecodeIngress* dflash2_host_ingress = nullptr;
    qwen3_6::DFlash2DecodeEgress* dflash2_host_egress = nullptr;
    std::optional<PinnedHostBuffer> dflash2_peer_host;
    qwen3_6::DFlash2DecodeIngress* dflash2_peer_host_ingress = nullptr;

    std::size_t workspace_logical_peak_bytes = 0;

    // See logits_capture's comment. Read-only; the caller owns thread-safety (single-lane, read
    // after wait() returns). Empty while capture is disabled.
    [[nodiscard]] std::span<const std::uint16_t> last_round_logits_bf16() const noexcept {
        return logits_capture;
    }

    // Turns the debug capture above on or off. Enabling sizes the host buffer to the round's
    // logits row count; disabling releases it. Must not be called while a round is in flight (the
    // executor holds its execution mutex across this call).
    void enable_logits_capture(bool enabled);

    // Debug-only, OFF by default: after each MTP decode round at tp == 2, read rank 1's MTP egress
    // back and compare it field for field with rank 0's. The two ranks run the acceptance Op over
    // bit-identical inputs, so their egress records are argued to agree; enabling this turns that
    // induction into a measurement. Costs one ~1 KiB device-to-host copy plus a host compare per
    // round while enabled, and nothing at all while off. Counters are cumulative over the
    // Program's lifetime; a mismatch is counted (per row, per field) rather than thrown, so a test
    // can read the totals after a clean run.
    void enable_peer_egress_check(bool enabled) noexcept;
    [[nodiscard]] std::uint64_t peer_egress_check_rounds() const noexcept {
        return peer_egress_rounds;
    }
    [[nodiscard]] std::uint64_t peer_egress_check_mismatches() const noexcept {
        return peer_egress_mismatches;
    }

private:
    void clear_lane(SequenceState& sequence, RequestControl& request) noexcept;
    // Moves a live retained sequence out of its lane and into a free registry entry, copying both
    // linear-attention states into the entry's dedicated slots. Leaves the lane empty (but does not
    // zero its state slots: the caller decides when that is safe).
    //
    // `allow_evict` is false for the displacement inside an attach. There the registry is full of
    // sessions a client may still return to, and the sequence asking for room is the one that just
    // lost its lane -- evicting a live entry to make room for it trades one session's prefix cache
    // for another's, which reads as flapping. The caller drops the displaced bundle instead.
    void detach_sequence(SequenceState& sequence, bool allow_evict = true);
    // Reverses `detach_sequence` onto `lane`. Displaces whatever that lane currently holds;
    // `displace_retained` says whether that bundle is a reusable sequence worth preserving.
    void attach_registry_sequence(std::int32_t index, SequenceState& sequence,
                                  bool displace_retained);
    // The entry a plan will attach, or nullptr when the plan uses its lane's own retained state.
    [[nodiscard]] const DetachedSequence*
    attached_entry(const RequestPlanImpl& plan) const noexcept;
    // Frees one entry (pages returned, slots become reusable). Returns false if the registry is
    // empty. `protect` is never evicted.
    bool evict_registry_entry(std::int32_t protect = -1) noexcept;
    // The least reusable entry other than `protect`, or -1 when none is left. This is the registry
    // capacity policy in one place: the shortest prefix goes first, and among equals the entry
    // detached longest ago.
    [[nodiscard]] std::int32_t least_reusable_entry(std::int32_t protect) const noexcept;
    // A page-holding cache bundle `plan_lane_admission` may spend, with the ranking that decides
    // which one goes first.
    struct ReclaimCandidate {
        runtime::AdmissionReclaim target;
        std::uint32_t text_pages    = 0;
        std::uint32_t backend_pages = 0;
        std::uint32_t frontier      = 0;
        std::uint64_t epoch         = 0;
    };
    void ordered_reset(SequenceState& sequence);
    void prepare_graphs();
    // Copies rank 0's counter lane onto rank 1. Called once, after prefill's bonus token is
    // sampled -- the only counter increment that happens on rank 0 and not on rank 1.
    [[nodiscard]] static Tensor token_counts_lane(const Tensor& storage, std::uint32_t lane);
    void publish_peer_token_counts(const SequenceState& sequence);
    // Mirrors `mtp_host_ingress` into `mtp_peer_host_ingress`, swapping every row's counter
    // pointer for rank 1's. No-op at tp1 or without MTP.
    void publish_peer_mtp_ingress(std::span<const std::uint32_t> lanes);
    void publish_peer_dflash2_ingress(std::span<const std::uint32_t> lanes);
    // Mirrors `ordinary_host_ingress` into `ordinary_peer_host_ingress` with every row's counter
    // pointer nulled. No-op at tp1 or without an ordinary frame.
    void publish_peer_ordinary_ingress();
    // Debug-only: reads rank 1's MTP egress back and compares it, field for field, with rank 0's.
    // No-op unless the check is enabled and a peer exists.
    void check_peer_mtp_egress(std::size_t rows);
    void install_sampling(SequenceState& sequence, RequestControl& request,
                          const ops::SamplingConfig& config);
    void set_device_i32(Tensor& tensor, std::int32_t value);
    void set_peer_i32(Tensor& tensor, std::int32_t value);
    void copy_tail(SequenceState& sequence, const Tensor& source);
    void copy_round_token();
    void copy_round_logits();
    void resolve_non_speculative_pending(SequenceState& sequence, RequestControl& request,
                                         std::uint32_t accepted_tokens, bool terminal);
    [[nodiscard]] runtime::PrefillStepResult advance_prefill(SequenceState& sequence,
                                                             RequestControl& request);
    void enqueue_dflash_context_append(std::span<const std::uint32_t> lanes,
                                       std::span<const std::uint32_t> starts,
                                       std::span<const std::uint32_t> counts);
    void validate_licensed_tokens(std::span<const TokenId> tokens) const;
    void mark_workspace_usage(std::size_t phase_bytes) noexcept;
    [[nodiscard]] runtime::BatchedGeneratedRound
    decode_ordinary_batch(std::span<const std::uint32_t> lanes,
                          std::span<const runtime::RoundBudget> budgets);
    [[nodiscard]] runtime::BatchedGeneratedRound
    decode_mtp_batch(std::span<const std::uint32_t> lanes,
                     std::span<const runtime::RoundBudget> budgets);
    [[nodiscard]] runtime::BatchedGeneratedRound
    decode_dflash_batch(std::span<const std::uint32_t> lanes,
                        std::span<const runtime::RoundBudget> budgets);
    [[nodiscard]] runtime::BatchedGeneratedRound
    decode_dflash2_batch(std::span<const std::uint32_t> lanes,
                         std::span<const runtime::RoundBudget> budgets);
    void reserve_sequence_kv(SequenceState& sequence, std::uint32_t text_pages,
                             std::uint32_t backend_pages);
    void resize_sequence_kv_entitlement(SequenceState& sequence, std::uint32_t text_pages,
                                        std::uint32_t backend_pages);
    void bind_sequence_kv(SequenceState& sequence);
    void unbind_sequence_kv(SequenceState& sequence) noexcept;
    void materialize_sequence_kv(SequenceState& sequence, std::uint32_t main_tokens,
                                 std::uint32_t backend_tokens = 0);
    void trim_sequence_kv(SequenceState& sequence, std::uint32_t main_tokens,
                          std::uint32_t backend_tokens = 0);
    void release_sequence_growth_entitlement(SequenceState& sequence) noexcept;
    [[nodiscard]] qwen3_6::PagedKVCache* backend_kv_cache() noexcept;
    [[nodiscard]] const qwen3_6::PagedKVCache* backend_kv_cache() const noexcept;
    [[nodiscard]] std::uint32_t backend_kv_valid(const SequenceState& sequence) const noexcept;
    [[nodiscard]] qwen3_6::PagedKVCacheView text_kv_view(const SequenceState& sequence) const;
    [[nodiscard]] qwen3_6::PagedKVCacheView mtp_kv_view(const SequenceState& sequence) const;
    [[nodiscard]] qwen3_6::PagedKVCacheView mtp_kv_view_peer(const SequenceState& sequence) const;
};

} // namespace ninfer::targets::qwen3_6::detail::NINFER_QWEN36_RUNTIME_NS

namespace ninfer::targets::qwen3_6::detail {

template <>
class ProgramImpl<NINFER_QWEN36_VARIANT> final : public NINFER_QWEN36_RUNTIME_NS::ProgramImplCore {
public:
    using NINFER_QWEN36_RUNTIME_NS::ProgramImplCore::ProgramImplCore;
};

} // namespace ninfer::targets::qwen3_6::detail
