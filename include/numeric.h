#ifndef _NUMERIC_H_
#define _NUMERIC_H_
#include "symbolic.h"
#include <stdexcept>
#include <string>

using namespace std;

enum class UpdateStrategy { RightLooking, LeftLooking };
enum class SchedulingStrategy { Level, SynchronizationFree };

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
                                         SchedulingStrategy scheduling)
{
    if (strategy != UpdateStrategy::RightLooking && strategy != UpdateStrategy::LeftLooking)
        throw std::invalid_argument("Invalid numeric update strategy");
    if (scheduling != SchedulingStrategy::Level &&
        scheduling != SchedulingStrategy::SynchronizationFree)
        throw std::invalid_argument("Invalid numeric scheduling strategy");
    if (scheduling == SchedulingStrategy::SynchronizationFree &&
        strategy != UpdateStrategy::LeftLooking)
        throw std::invalid_argument("synchronization-free scheduling requires left-looking updates");
}

void LUonDevice(Symbolic_Matrix &, ostream &, ostream &, bool,
                UpdateStrategy = UpdateStrategy::RightLooking,
                SchedulingStrategy = SchedulingStrategy::Level,
                unsigned sf_workers = 0);

#endif
