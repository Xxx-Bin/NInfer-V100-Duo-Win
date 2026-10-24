#pragma once

#include <cstdint>
#include <limits>
#include <stdexcept>

namespace ninfer::targets::qwen3_6::detail::NINFER_QWEN36_RUNTIME_NS {

/**
 * Qwen3.6's target-local mapping from a request lane (or a detached prefix-registry entry) to its
 * linear-attention state slots.
 *
 * Lane slots are the two the executor has always used: `lane` holds the running continuation, and
 * `max_concurrency + lane` holds the single rewrite checkpoint of the sequence that lane retains
 * after its request completes.
 *
 * Registry slots exist so a sequence can outlive the lane that produced it. They are appended after
 * the lane range and are addressed by registry entry index, never by lane, which is what keeps the
 * two ranges disjoint: a lane can be handed to an unrelated request without its detached state
 * being overwritten, and a detached entry can be re-attached to any lane.
 */
struct LinearStateSlots {
    [[nodiscard]] static std::int32_t state_slot_count(std::uint32_t max_concurrency,
                                                       std::uint32_t registry_entries = 0,
                                                       bool drop_current = false) {
        if (max_concurrency == 0 ||
            max_concurrency > static_cast<std::uint32_t>(max_concurrency_limit())) {
            throw std::invalid_argument("Qwen3.6 Linear Attention concurrency is invalid");
        }
        if (registry_entries > static_cast<std::uint32_t>(registry_entry_limit())) {
            throw std::invalid_argument("Qwen3.6 prefix registry entry count is invalid");
        }
        const std::int32_t lane_slots =
            static_cast<std::int32_t>(2U * max_concurrency);
        // One slot per entry when the entry keeps only its rewrite checkpoint.
        const std::int32_t registry_slots =
            static_cast<std::int32_t>((drop_current ? 1U : 2U) * registry_entries);
        if (lane_slots > std::numeric_limits<std::int32_t>::max() - registry_slots) {
            throw std::invalid_argument("Qwen3.6 Linear Attention slot count overflows");
        }
        return lane_slots + registry_slots;
    }

    [[nodiscard]] static std::int32_t max_concurrency_limit() {
        return std::numeric_limits<std::int32_t>::max() / 4;
    }

    [[nodiscard]] static std::int32_t registry_entry_limit() {
        return std::numeric_limits<std::int32_t>::max() / 4;
    }

    [[nodiscard]] static std::int32_t current_state_slot(std::uint32_t lane,
                                                         std::uint32_t max_concurrency) {
        if (lane >= max_concurrency) {
            throw std::out_of_range("Qwen3.6 Linear Attention lane is out of range");
        }
        return static_cast<std::int32_t>(lane);
    }

    [[nodiscard]] static std::int32_t rewrite_checkpoint_state_slot(std::uint32_t lane,
                                                                    std::uint32_t max_concurrency) {
        return static_cast<std::int32_t>(max_concurrency) +
               current_state_slot(lane, max_concurrency);
    }

    /**
     * First slot of registry entry `entry`. A detached entry keeps BOTH of the sequence's states,
     * because either reuse path may select it later: `AppendAtFrontier` needs the continuation at
     * the execution frontier, and the checkpoint restores need the state at the rewrite boundary.
     *
     * Under `drop_current` an entry keeps ONLY the rewrite checkpoint, so there is no current slot
     * to name: the index this function would return belongs to a different entry's checkpoint. It
     * throws instead of returning an index that silently points at somebody else's state.
     */
    [[nodiscard]] static std::int32_t registry_state_slot(std::uint32_t entry,
                                                          std::uint32_t max_concurrency,
                                                          bool drop_current = false) {
        if (drop_current) { throw std::logic_error("registry keeps no current slot"); }
        return static_cast<std::int32_t>(2U * max_concurrency) + checked_entry_offset(entry, 2U);
    }

    [[nodiscard]] static std::int32_t
    registry_checkpoint_state_slot(std::uint32_t entry, std::uint32_t max_concurrency,
                                   bool drop_current = false) {
        return static_cast<std::int32_t>(2U * max_concurrency) +
               checked_entry_offset(entry, drop_current ? 1U : 2U) + (drop_current ? 0 : 1);
    }

private:
    [[nodiscard]] static std::int32_t checked_entry_offset(std::uint32_t entry,
                                                           std::uint32_t stride) {
        if (entry >= static_cast<std::uint32_t>(registry_entry_limit())) {
            throw std::out_of_range("Qwen3.6 prefix registry entry is out of range");
        }
        return static_cast<std::int32_t>(stride * entry);
    }
};

} // namespace ninfer::targets::qwen3_6::detail::NINFER_QWEN36_RUNTIME_NS
