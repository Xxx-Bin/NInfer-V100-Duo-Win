#pragma once

#include "ninfer/types.h"

#include <cstddef>
#include <cstdint>
#include <span>
#include <vector>

namespace ninfer::runtime {

using ::ninfer::FinishReason;
using ::ninfer::KvCapacityMode;
using ::ninfer::KvCapacityPolicy;
using ::ninfer::OutputChannel;
using ::ninfer::ResolvedSamplingParameters;
using ::ninfer::StopPolicy;
using ::ninfer::StopString;
using ::ninfer::TokenId;

// Engine has already selected the registered model/mode preset, applied every explicit override,
// and validated these values before constructing the runtime request.
struct ResolvedExecutionOptions {
    ResolvedSamplingParameters sampling;
    std::uint32_t requested_output_tokens = 0;
    bool allow_prefix_reuse               = true;
};

struct ResolvedRequestOptions {
    ResolvedExecutionOptions execution;
    StopPolicy stop;
    OutputOptions output;
};

struct OutputDecision {
    std::uint32_t accepted_tokens = 0;
    FinishReason finish_reason    = FinishReason::None;

    [[nodiscard]] bool finished() const noexcept { return finish_reason != FinishReason::None; }
};

// Complete request-lifetime ownership in the three independently exhausted admission domains.
// Values are already rounded to the physical allocation granularity by the target.
struct AdmissionResources {
    std::uint32_t active_lanes     = 0;
    std::uint32_t main_kv_pages    = 0;
    std::uint32_t backend_kv_pages = 0;
};

// A cache bundle a blocked admission gives up so that its plan fits. The target that owns the
// bundle defines what each kind costs; the Engine only applies the list and invalidates the plans
// that named it.
enum class AdmissionReclaimKind : std::uint8_t {
    // The retained sequence parked on this lane slot. Releasing it destroys the cached prefix.
    RetainedSequence,
    // A displaced prefix held outside every lane by the target's prefix registry. Evicting it
    // destroys the cached prefix.
    PrefixEntry,
};

struct AdmissionReclaim {
    AdmissionReclaimKind kind = AdmissionReclaimKind::RetainedSequence;
    // Lane slot or prefix-registry entry index, in the space of `kind`.
    std::uint32_t slot = 0;
};

// The target's answer to "can this request plan take this lane, and what must be given up first".
// `reclaim` is ordered by execution and already minimal: applying it in order is exactly the state
// the feasibility test was computed against, so `feasible` never disagrees with the reservation
// that follows. An infeasible plan carries no reclaim list.
struct AdmissionPlan {
    bool feasible = false;
    std::vector<AdmissionReclaim> reclaim;
};

struct RequestPlanSummary {
    std::uint32_t prompt_tokens           = 0;
    std::uint32_t reusable_prompt_tokens  = 0;
    std::uint32_t requested_output_tokens = 0;
    std::uint32_t effective_output_tokens = 0;
    FinishReason effective_limit_reason   = FinishReason::None;
    std::size_t transient_bytes           = 0;
    std::size_t transient_alignment       = 1;
    AdmissionResources admission;
    std::uint64_t service_work_quanta = 0;
};

struct BeginSummary {
    std::uint32_t prompt_tokens        = 0;
    std::uint32_t reused_prompt_tokens = 0;
    PrefixReusePath prefix_reuse_path  = PrefixReusePath::FullReset;
    // Probe only (no control flow reads this): the reuse base was a prefix-registry entry.
    bool attached_from_registry = false;
};

struct GeneratedRound {
    std::span<const TokenId> tokens;
};

struct BatchedGeneratedRound {
    std::span<const TokenId> tokens;
    std::span<const std::int32_t> row_counts;
    std::uint32_t row_stride = 1;
};

struct PrefillStepResult {
    BeginSummary summary;
    GeneratedRound round;
    std::uint32_t processed_prompt_tokens = 0;
    bool complete                         = false;
};

struct RoundBudget {
    std::uint32_t generated_tokens_remaining = 0;
};

// Target-produced affine reservation curve for one Main KV physical-capacity axis. The byte
// values come from complete target physical layout plans, not from a model geometry formula in
// the common runtime.
struct SequenceCapacityCurve {
    std::uint32_t main_page_tokens                   = 0;
    std::uint32_t minimum_main_page_groups           = 0;
    std::uint32_t maximum_main_page_groups           = 0;
    std::size_t minimum_device_reservation_bytes     = 0;
    std::size_t bytes_per_additional_main_page_group = 0;

    [[nodiscard]] std::size_t reservation_bytes(std::uint32_t main_page_groups) const;
    [[nodiscard]] std::uint32_t resolved_tokens(std::uint32_t main_page_groups) const;
};

struct KvCapacityResolution {
    KvCapacityMode mode                              = KvCapacityMode::Explicit;
    std::uint32_t main_page_groups                   = 0;
    std::uint32_t maximum_main_page_groups           = 0;
    std::uint32_t resolved_tokens                    = 0;
    std::size_t minimum_runtime_reservation_bytes    = 0;
    std::size_t bytes_per_additional_main_page_group = 0;
    std::size_t runtime_reservation_bytes            = 0;
    std::size_t available_after_weights_bytes        = 0;
    std::size_t available_after_startup_bytes        = 0;
    std::size_t automatic_headroom_bytes             = 0;
    std::size_t planned_slack_bytes                  = 0;
};

} // namespace ninfer::runtime
