#include "targets/qwen3_6/impl/runtime/instance.h"
#include "targets/qwen3_6/impl/runtime/program.h"

#include "targets/qwen3_6/impl/runtime/schedule.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>

namespace ninfer::targets::qwen3_6::detail::NINFER_QWEN36_RUNTIME_NS {
namespace {

void validate_sampling(const ResolvedSamplingParameters& sampling) {
    if (!std::isfinite(sampling.temperature) || !std::isfinite(sampling.top_p) ||
        !std::isfinite(sampling.min_p) || !std::isfinite(sampling.presence_penalty) ||
        !std::isfinite(sampling.frequency_penalty)) {
        throw std::invalid_argument("sampling parameters must be finite");
    }
    if (sampling.top_p < 0.0F || sampling.top_p > 1.0F) {
        throw std::invalid_argument("top_p must be in [0,1]");
    }
    if (sampling.min_p < 0.0F || sampling.min_p > 1.0F) {
        throw std::invalid_argument("min_p must be in [0,1]");
    }
}

ops::SamplingConfig translate_sampling(const ResolvedSamplingParameters& source) {
    ops::SamplingConfig out;
    out.temperature       = source.temperature;
    out.top_k             = source.top_k;
    out.top_p             = source.top_p;
    out.min_p             = source.min_p;
    out.presence_penalty  = source.presence_penalty;
    out.frequency_penalty = source.frequency_penalty;
    out.seed              = source.seed;
    out.token_counts      = nullptr;
    return out;
}

std::uint32_t pages_for_tokens(std::uint32_t tokens) noexcept {
    return 1U + (tokens - 1U) / static_cast<std::uint32_t>(kPagedKVPageSize);
}

std::uint64_t projected_service_work(const runtime::RequestPlanSummary& summary,
                                     std::uint32_t reuse_base, std::uint32_t prefill_chunk,
                                     std::size_t prefill_splits) noexcept {
    const std::uint32_t suffix = summary.prompt_tokens - reuse_base;
    const std::uint64_t prefill_units =
        suffix == 0
            ? 1ULL
            : 1ULL + (static_cast<std::uint64_t>(suffix) - 1ULL) / prefill_chunk + prefill_splits;
    const std::uint64_t decode_units =
        summary.effective_output_tokens == 0 ? 0ULL : summary.effective_output_tokens - 1ULL;
    return prefill_units + decode_units;
}

} // namespace

RequestBasePlan
ProgramImplCore::plan_request_base(const PreparedPromptData& prompt,
                                   const runtime::ResolvedExecutionOptions& options) {
    if (prompt.token_ids.empty()) { throw std::invalid_argument("prompt must contain tokens"); }
    if (prompt.token_ids.size() > capacity) {
        throw std::invalid_argument("prompt exceeds configured context capacity");
    }
    if (prompt.token_ids.size() > std::numeric_limits<std::uint32_t>::max()) {
        throw std::overflow_error("prompt token count exceeds uint32");
    }
    for (const TokenId id : prompt.token_ids) {
        if (id < 0 || id >= TextConfig::token_domain) {
            throw std::invalid_argument("prompt contains token outside the 248077-token domain");
        }
    }
    if (prompt.token_types.size() != prompt.token_ids.size() ||
        prompt.positions.size() != 3ULL * prompt.token_ids.size()) {
        throw std::invalid_argument("prepared prompt token metadata has an invalid shape");
    }
    if (prompt.has_media() != !prompt.media_payloads.empty() ||
        prompt.media_payloads.size() != prompt.vision_items.size()) {
        throw std::invalid_argument("prepared prompt media payload is incomplete");
    }
    for (std::size_t i = 0; i < prompt.media_payloads.size(); ++i) {
        if (!prompt.media_payloads[i] ||
            prompt.media_payloads[i]->patch_elements !=
                prompt.vision_items[i].patch_count * kPreparedVisionPatchFeatures) {
            throw std::invalid_argument("prepared prompt media item payload has an invalid shape");
        }
    }
    if (prompt.has_media() && !vision_enabled) {
        throw std::invalid_argument("Vision is disabled for this Engine");
    }
    validate_sampling(options.sampling);

    auto base                             = std::make_unique<RequestBasePlanImpl>();
    base->summary.prompt_tokens           = static_cast<std::uint32_t>(prompt.token_ids.size());
    base->summary.requested_output_tokens = options.requested_output_tokens;
    const std::uint32_t capacity_output =
        capacity - base->summary.prompt_tokens + static_cast<std::uint32_t>(1);
    base->summary.effective_output_tokens =
        std::min(options.requested_output_tokens, capacity_output);
    base->summary.effective_limit_reason = options.requested_output_tokens <= capacity_output
                                               ? FinishReason::OutputLimit
                                               : FinishReason::ContextCapacity;
    base->summary.transient_alignment    = 1;
    base->summary.transient_bytes        = 0;
    base->sampling                       = translate_sampling(options.sampling);
    base->allow_prefix_reuse             = options.allow_prefix_reuse;
    const std::uint32_t reserved_context_tokens =
        base->summary.prompt_tokens + (base->summary.effective_output_tokens == 0
                                           ? 0U
                                           : base->summary.effective_output_tokens - 1U);
    base->text_kv_page_entitlement = pages_for_tokens(reserved_context_tokens);
    if (speculative_backend == SpeculativeBackend::Mtp) {
        const std::uint32_t mtp_tokens    = static_cast<std::uint32_t>(std::min<std::uint64_t>(
            capacity, static_cast<std::uint64_t>(reserved_context_tokens) + draft_window - 1ULL));
        base->backend_kv_page_entitlement = pages_for_tokens(mtp_tokens);
    } else if (speculative_backend == SpeculativeBackend::DFlash) {
        base->backend_kv_page_entitlement = pages_for_tokens(reserved_context_tokens);
    }
    base->summary.admission = runtime::AdmissionResources{
        .active_lanes     = 1,
        .main_kv_pages    = base->text_kv_page_entitlement,
        .backend_kv_pages = base->backend_kv_page_entitlement,
    };
    if (prompt.has_media()) {
        auto control =
            std::make_shared<qwen3_6::VisionControl>(qwen3_6::build_vision_control(prompt));
        std::size_t max_merged     = 0;
        std::uint32_t previous_end = 0;
        for (const qwen3_6::VisionItemControl& item : control->items) {
            if (item.scatter_indices.empty()) {
                throw std::invalid_argument("vision item has no Text consumer columns");
            }
            const auto first = static_cast<std::uint32_t>(item.scatter_indices.front());
            const auto last  = static_cast<std::uint32_t>(item.scatter_indices.back());
            const std::uint32_t begin =
                speculative_backend == SpeculativeBackend::Mtp && first != 0 ? first - 1 : first;
            const std::uint32_t end = last + 1;
            if (begin < previous_end) {
                throw std::invalid_argument("vision item consumer spans overlap");
            }
            if (end > base->summary.prompt_tokens) {
                throw std::invalid_argument("vision item consumer span exceeds prompt");
            }
            if (schedule::VisionContext::workspace_bytes(item) > work.capacity()) {
                throw std::invalid_argument("vision item exceeds the Program workspace envelope");
            }
            previous_end = end;
            max_merged   = std::max(max_merged, item.merged_count);
        }
        base->vision_transient_bytes = schedule::VisionContext::output_transient_bytes(max_merged);
        base->vision_control         = std::move(control);
    }

    if (prompt.identity.rewrite_checkpoint) {
        const RewriteCheckpointSpec candidate = *prompt.identity.rewrite_checkpoint;
        if (candidate.frontier == 0 || candidate.frontier > base->summary.prompt_tokens) {
            throw std::invalid_argument(
                "rewrite checkpoint frontier must lie at or inside the prompt frontier");
        }
        base->rewrite_checkpoint = candidate;
    }
    const std::size_t cold_prefill_splits =
        (base->vision_control != nullptr ? base->vision_control->items.size() : 0ULL) +
        (base->rewrite_checkpoint &&
                 base->rewrite_checkpoint->frontier < base->summary.prompt_tokens
             ? 1ULL
             : 0ULL);
    base->summary.service_work_quanta =
        projected_service_work(base->summary, 0, prefill_chunk, cold_prefill_splits);
    return RequestBasePlan(std::move(base));
}

RequestPlan ProgramImplCore::plan_request_for_lane(std::uint32_t lane,
                                                   const PreparedPromptData& prompt,
                                                   const RequestBasePlan& base_plan) {
    if (lane >= max_concurrency) { throw std::out_of_range("request lane is out of range"); }
    const RequestControl& request = requests[lane];
    const SequenceState& sequence = sequences[lane];
    if (request.lifecycle == Lifecycle::Prefilling || request.lifecycle == Lifecycle::Active ||
        request.lifecycle == Lifecycle::Pending) {
        throw std::logic_error("cannot plan a request while Program is active or pending");
    }
    if (base_plan.impl_ == nullptr) { throw std::logic_error("request base plan is empty"); }
    const RequestBasePlanImpl& base = *base_plan.impl_;

    auto plan                         = std::make_unique<RequestPlanImpl>();
    plan->summary                     = base.summary;
    plan->sampling                    = base.sampling;
    plan->text_kv_page_entitlement    = base.text_kv_page_entitlement;
    plan->backend_kv_page_entitlement = base.backend_kv_page_entitlement;

    // The reuse base is either this lane's own retained sequence or a detached registry entry.
    // Both are offered to the same two decisions below (append at frontier, restore a checkpoint);
    // only the owner differs, and a registry base additionally has to be attached before prefill.
    const DetachedSequence* attach = nullptr;
    if (base.allow_prefix_reuse && prompt.identity.reusable) {
        if (sequence.retained) {
            const bool dflash_append_ready =
                speculative_backend != SpeculativeBackend::DFlash ||
                sequence.dflash_context_frontier == sequence.execution_frontier;
            if (sequence.execution_frontier != 0 && dflash_append_ready &&
                qwen3_6::detail::prefix_matches(prompt, sequence.ledger, sequence.prefix_identity,
                                                sequence.execution_frontier)) {
                plan->reuse      = ReusePath::AppendAtFrontier;
                plan->reuse_base = sequence.execution_frontier;
            } else if (sequence.rewrite_checkpoint.valid &&
                       sequence.rewrite_checkpoint.frontier != 0 &&
                       sequence.rewrite_checkpoint.frontier <= prompt.token_ids.size() &&
                       qwen3_6::detail::prefix_matches(prompt, sequence.ledger,
                                                       sequence.prefix_identity,
                                                       sequence.rewrite_checkpoint.frontier)) {
                plan->reuse      = restore_path(sequence.rewrite_checkpoint.kind);
                plan->reuse_base = sequence.rewrite_checkpoint.frontier;
            }
        }
        if (plan->reuse == ReusePath::FullReset && !registry.empty()) {
            // Append candidates win over checkpoint candidates across the whole registry, and
            // within a class the longest prefix wins, so the choice does not depend on slot order.
            //
            // Under `registry_drops_current` no entry keeps its continuation state, so there is no
            // append candidate at all: pass 0 is skipped whole rather than evaluated and rejected,
            // which is what keeps `attach && AppendAtFrontier` unreachable by construction.
            std::size_t best    = registry.size();
            for (std::size_t pass = registry_drops_current ? 1U : 0U;
                 pass < 2 && best == registry.size(); ++pass) {
                for (std::size_t i = 0; i < registry.size(); ++i) {
                    const std::optional<DetachedSequence>& entry = registry[i];
                    if (!entry) { continue; }
                    const bool dflash_append_ready =
                        speculative_backend != SpeculativeBackend::DFlash ||
                        entry->dflash_context_frontier == entry->execution_frontier;
                    std::uint32_t frontier = 0;
                    ReusePath path         = ReusePath::FullReset;
                    if (pass == 0) {
                        if (entry->execution_frontier != 0 && dflash_append_ready &&
                            qwen3_6::detail::prefix_matches(prompt, entry->ledger,
                                                            entry->prefix_identity,
                                                            entry->execution_frontier)) {
                            frontier = entry->execution_frontier;
                            path     = ReusePath::AppendAtFrontier;
                        }
                    } else if (entry->rewrite_checkpoint.valid &&
                               entry->rewrite_checkpoint.frontier != 0 &&
                               entry->rewrite_checkpoint.frontier <= prompt.token_ids.size() &&
                               qwen3_6::detail::prefix_matches(
                                   prompt, entry->ledger, entry->prefix_identity,
                                   entry->rewrite_checkpoint.frontier)) {
                        frontier = entry->rewrite_checkpoint.frontier;
                        path     = restore_path(entry->rewrite_checkpoint.kind);
                    }
                    if (path == ReusePath::FullReset) { continue; }
                    if (best == registry.size() || frontier > plan->reuse_base) {
                        best            = i;
                        plan->reuse      = path;
                        plan->reuse_base = frontier;
                    }
                }
            }
            if (best != registry.size()) {
                attach                     = &*registry[best];
                plan->attach_registry_index = static_cast<std::int32_t>(best);
            }
        }
    }

    const bool reuse_tail_hidden_valid =
        attach != nullptr ? attach->tail_hidden_valid : sequence.tail_hidden_valid;
    const std::uint32_t reuse_mtp_kv_valid =
        attach != nullptr ? attach->mtp_kv_valid : sequence.mtp_kv_valid;
    const std::uint32_t reuse_dflash_frontier =
        attach != nullptr ? attach->dflash_context_frontier : sequence.dflash_context_frontier;
    const bool reuse_has_backend =
        attach != nullptr ? attach->kv.backend.has_value()
                          : (sequence.kv && sequence.kv->backend.has_value());
    const std::vector<TokenId>& reuse_ledger =
        attach != nullptr ? attach->ledger : sequence.ledger;
    const qwen3_6::detail::ResidentPrefixIdentity& reuse_identity =
        attach != nullptr ? attach->prefix_identity : sequence.prefix_identity;
    const RewriteCheckpoint& reuse_checkpoint =
        attach != nullptr ? attach->rewrite_checkpoint : sequence.rewrite_checkpoint;

    if (speculative_backend == SpeculativeBackend::Mtp) {
        const bool append_ready =
            plan->reuse == ReusePath::AppendAtFrontier && reuse_tail_hidden_valid &&
            decoder->mtp_cache() != nullptr &&
            (plan->reuse_base == 0 || reuse_mtp_kv_valid >= plan->reuse_base - 1);
        const bool checkpoint_ready = is_rewrite_checkpoint_restore(plan->reuse) &&
                                      decoder->mtp_cache() != nullptr && plan->reuse_base != 0 &&
                                      reuse_mtp_kv_valid >= plan->reuse_base - 1;
        if (plan->reuse != ReusePath::FullReset && !append_ready && !checkpoint_ready) {
            plan->reuse                 = ReusePath::FullReset;
            plan->reuse_base            = 0;
            plan->attach_registry_index = -1;
            attach                      = nullptr;
        }
    }

    if (is_rewrite_checkpoint_restore(plan->reuse) &&
        speculative_backend == SpeculativeBackend::DFlash &&
        (!dflash || !reuse_has_backend || reuse_dflash_frontier < plan->reuse_base)) {
        plan->reuse                 = ReusePath::FullReset;
        plan->reuse_base            = 0;
        plan->attach_registry_index = -1;
    }

    const std::optional<RewriteCheckpointSpec>& desired = base.rewrite_checkpoint;
    const bool existing_checkpoint_matches =
        desired && plan->reuse != ReusePath::FullReset && reuse_checkpoint.valid &&
        reuse_checkpoint.frontier == desired->frontier &&
        qwen3_6::detail::prefix_matches(prompt, reuse_ledger, reuse_identity, desired->frontier);
    if (!desired) {
        plan->rewrite_checkpoint_action = RewriteCheckpointAction::Drop;
    } else if (existing_checkpoint_matches) {
        plan->rewrite_checkpoint_action = reuse_checkpoint.kind == desired->kind
                                              ? RewriteCheckpointAction::KeepExisting
                                              : RewriteCheckpointAction::ReclassifyExisting;
    } else if (desired->frontier > plan->reuse_base) {
        plan->rewrite_checkpoint_action  = RewriteCheckpointAction::CaptureNew;
        plan->rewrite_checkpoint_capture = desired;
    } else {
        // The selected continuation state is already past the desired boundary. It remains a
        // valid hit; do not replay an otherwise reusable prefix merely to materialize an older
        // auxiliary snapshot. A later request can still use the checkpoint currently retained.
        plan->rewrite_checkpoint_action = RewriteCheckpointAction::DeferCapture;
    }

    plan->summary.reusable_prompt_tokens = plan->reuse_base;
    if (speculative_backend == SpeculativeBackend::Mtp) {
        if (plan->reuse == ReusePath::FullReset) {
            plan->prepare_mtp = true;
        } else if (plan->reuse == ReusePath::AppendAtFrontier) {
            plan->prepare_mtp = true;
            plan->mtp_bridge  = plan->reuse_base < plan->summary.prompt_tokens
                                    ? MtpBridgeMode::BeforeSuffix
                                    : MtpBridgeMode::AfterExactHit;
        } else if (is_rewrite_checkpoint_restore(plan->reuse)) {
            plan->prepare_mtp = true;
            plan->mtp_bridge  = plan->reuse_base < plan->summary.prompt_tokens
                                    ? MtpBridgeMode::BeforeSuffix
                                    : MtpBridgeMode::AfterExactHit;
        }
    }

    if (base.vision_control != nullptr) {
        VisionPrefillPlan vision;
        vision.control = base.vision_control;
        vision.uses.reserve(base.vision_control->items.size());
        for (std::size_t index = 0; index < base.vision_control->items.size(); ++index) {
            const qwen3_6::VisionItemControl& item = base.vision_control->items[index];
            const auto first          = static_cast<std::uint32_t>(item.scatter_indices.front());
            const auto last           = static_cast<std::uint32_t>(item.scatter_indices.back());
            const std::uint32_t begin = plan->prepare_mtp && first != 0 ? first - 1 : first;
            const std::uint32_t end   = last + 1;
            if (end <= plan->reuse_base) { continue; }
            vision.uses.push_back(VisionUseSpan{begin, end, static_cast<std::uint32_t>(index)});
        }
        if (!vision.uses.empty()) {
            plan->summary.transient_alignment = 256;
            plan->summary.transient_bytes     = base.vision_transient_bytes;
            plan->vision                      = std::move(vision);
        }
    }

    const std::size_t prefill_splits =
        (plan->vision ? plan->vision->uses.size() : 0ULL) +
        (plan->rewrite_checkpoint_capture &&
                 plan->rewrite_checkpoint_capture->frontier < plan->summary.prompt_tokens
             ? 1ULL
             : 0ULL);
    plan->summary.service_work_quanta =
        projected_service_work(plan->summary, plan->reuse_base, prefill_chunk, prefill_splits);
    return RequestPlan(std::move(plan));
}

} // namespace ninfer::targets::qwen3_6::detail::NINFER_QWEN36_RUNTIME_NS
