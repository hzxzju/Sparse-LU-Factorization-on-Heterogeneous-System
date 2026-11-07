#ifndef GLU_NUMERIC_SM_PADDING_H
#define GLU_NUMERIC_SM_PADDING_H

#include <cstddef>
#include <limits>
#include <stdexcept>

// Host-side resource plan for the opt-in numeric SM activity experiment.
// Runtime occupancy must additionally confirm exactly one resident block/SM.
struct NumericSmPaddingPlan {
    unsigned grid_blocks;
    std::size_t dynamic_shared_bytes;
};

inline NumericSmPaddingPlan MakeNumericSmPaddingPlan(
        bool cooperative_launch, int sm_count, int max_grid_x,
        std::size_t shared_per_sm, std::size_t shared_per_block,
        std::size_t static_shared_bytes)
{
    if (!cooperative_launch)
        throw std::invalid_argument("numeric SM padding requires cooperative launch support");
    if (sm_count <= 0 || max_grid_x < sm_count)
        throw std::invalid_argument("numeric SM padding requires a valid one-block-per-SM grid");
    if (!shared_per_sm || shared_per_sm / 2 > std::numeric_limits<std::size_t>::max() - 256)
        throw std::invalid_argument("numeric SM padding has invalid shared memory limits");
    // A small margin above half the SM capacity prevents two resident blocks,
    // including when the kernel's static shared memory is very small.
    const std::size_t dynamic = shared_per_sm / 2 + 256;
    if (static_shared_bytes > shared_per_block ||
        dynamic > shared_per_block - static_shared_bytes ||
        static_shared_bytes > shared_per_sm || dynamic > shared_per_sm - static_shared_bytes)
        throw std::invalid_argument("numeric SM padding cannot reserve half-SM shared memory per block");
    return NumericSmPaddingPlan{static_cast<unsigned>(sm_count), dynamic};
}

#endif
