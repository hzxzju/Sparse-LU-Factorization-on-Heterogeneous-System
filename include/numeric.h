#ifndef _NUMERIC_H_
#define _NUMERIC_H_
#include "symbolic.h"
#include "stage_timing.h"
#include <stdexcept>
#include <string>
#include <functional>

using namespace std;

enum class UpdateStrategy { RightLooking, LeftLooking };
enum class SchedulingStrategy { Level, SynchronizationFree };
enum class SFUpdateStrategy { Block, Wave };

inline SFUpdateStrategy ParseSFUpdateStrategy(const std::string &name)
{
    if (name == "block") return SFUpdateStrategy::Block;
    if (name == "wave") return SFUpdateStrategy::Wave;
    throw std::invalid_argument("sf_update must be 'block' or 'wave'");
}

inline const char *SFUpdateStrategyName(SFUpdateStrategy strategy)
{
    return strategy == SFUpdateStrategy::Wave ? "wave" : "block";
}

inline UpdateStrategy ParseUpdateStrategy(const std::string &name)
{
    if (name == "right-looking" || name == "right" || name == "rl")
        return UpdateStrategy::RightLooking;
    if (name == "left-looking" || name == "left" || name == "ll")
        return UpdateStrategy::LeftLooking;
    throw std::invalid_argument(
        "update_strategy must be 'right-looking' or 'left-looking'");
}

inline const char *UpdateStrategyName(UpdateStrategy strategy)
{
    return strategy == UpdateStrategy::LeftLooking ? "left-looking" : "right-looking";
}

inline SchedulingStrategy ParseSchedulingStrategy(const std::string &name)
{
    if (name == "level")
        return SchedulingStrategy::Level;
    if (name == "synchronization-free")
        return SchedulingStrategy::SynchronizationFree;
    throw std::invalid_argument(
        "scheduling must be 'level' or 'synchronization-free'");
}

inline const char *SchedulingStrategyName(SchedulingStrategy scheduling)
{
    return scheduling == SchedulingStrategy::SynchronizationFree
        ? "synchronization-free" : "level";
}

inline void ValidateNumericConfiguration(UpdateStrategy strategy,
                                         SchedulingStrategy scheduling,
                                         SFUpdateStrategy sf_update = SFUpdateStrategy::Block)
{
    if (strategy != UpdateStrategy::RightLooking && strategy != UpdateStrategy::LeftLooking)
        throw std::invalid_argument("Invalid numeric update strategy");
    if (scheduling != SchedulingStrategy::Level &&
        scheduling != SchedulingStrategy::SynchronizationFree)
        throw std::invalid_argument("Invalid numeric scheduling strategy");
    if (scheduling == SchedulingStrategy::SynchronizationFree &&
        strategy != UpdateStrategy::LeftLooking)
        throw std::invalid_argument("synchronization-free scheduling requires left-looking updates");
    if (sf_update != SFUpdateStrategy::Block && sf_update != SFUpdateStrategy::Wave)
        throw std::invalid_argument("Invalid SF update strategy");
    if (sf_update == SFUpdateStrategy::Wave &&
        (strategy != UpdateStrategy::LeftLooking || scheduling != SchedulingStrategy::SynchronizationFree))
        throw std::invalid_argument("sf_update wave requires left-looking synchronization-free scheduling");
}

void LUonDevice(Symbolic_Matrix &, ostream &, ostream &, bool,
                UpdateStrategy = UpdateStrategy::LeftLooking,
                SchedulingStrategy = SchedulingStrategy::SynchronizationFree,
                unsigned sf_workers = 0, SFUpdateStrategy sf_update = SFUpdateStrategy::Block);

// One setup, optional warmup, then repeat factor/host-solve cycles. The callback
// sees downloaded factors and must finish the solve/refinement before returning.
void LUonDeviceRepeated(Symbolic_Matrix &, ostream &, ostream &, bool,
                        UpdateStrategy, SchedulingStrategy, unsigned sf_workers,
                        SFUpdateStrategy sf_update, unsigned repeat,
                        const std::function<void()> &solve);

// CLI-only final teardown; library callers retain their runtime context.
bool ShutdownGPUDevice(int device, ostream &errors);

// Keep setup timings and only the final cycle; discard the warmup and earlier
// cycles. All GPU work must be synchronized inside the iteration callback.
inline void RunNumericIterations(unsigned repeat, const std::function<void()> &iteration)
{
    if (!repeat) throw std::invalid_argument("repeat must be positive");
    GluTimings *timings = glu_timing::active();
    GluTimings baseline;
    if (timings) {
        glu_timing_switch(timings, GLU_ANALYSIS_OTHER);
        baseline = *timings;
    }
    const uint64_t cycles = uint64_t(repeat) + (repeat > 1 ? 1 : 0);
    for (uint64_t cycle = 0; cycle < cycles; ++cycle) {
        if (timings) {
            *timings = baseline;
            timings->boundary = glu_monotonic_now();
        }
        iteration();
    }
}

#endif
