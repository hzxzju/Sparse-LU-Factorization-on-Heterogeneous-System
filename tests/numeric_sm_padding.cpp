#include "numeric.h"
#include "numeric_sm_padding.h"
#include <iostream>
#include <stdexcept>

static void require(bool condition, const char *message)
{
    if (!condition) throw std::runtime_error(message);
}

template<class Function> static void rejects(Function function)
{
    bool rejected = false;
    try { function(); } catch (const std::invalid_argument &) { rejected = true; }
    require(rejected, "unsupported numeric SM padding configuration accepted");
}

int main()
{
    try {
        const auto plan = MakeNumericSmPaddingPlan(true, 104, 2147483647,
                                                   65536, 65536, 16);
        require(plan.grid_blocks == 104, "grid must contain one block per SM");
        require(plan.dynamic_shared_bytes == 33024, "unexpected X201 shared-memory reservation");
        require(2 * (plan.dynamic_shared_bytes + 16) > 65536,
                "two blocks could fit in one SM");
        require(plan.dynamic_shared_bytes + 16 <= 65536, "one block cannot fit");
        rejects([] { MakeNumericSmPaddingPlan(false, 104, 104, 65536, 65536, 16); });
        rejects([] { MakeNumericSmPaddingPlan(true, 0, 104, 65536, 65536, 16); });
        rejects([] { MakeNumericSmPaddingPlan(true, 104, 103, 65536, 65536, 16); });
        rejects([] { MakeNumericSmPaddingPlan(true, 104, 104, 0, 65536, 16); });
        rejects([] { MakeNumericSmPaddingPlan(true, 104, 104, 65536, 32768, 16); });
        rejects([] { MakeNumericSmPaddingPlan(true, 104, 104, 65536, 65536, 40000); });
        rejects([] { ValidateNumericConfiguration(UpdateStrategy::LeftLooking,
                                                   SchedulingStrategy::Level, true); });
        rejects([] { ValidateNumericConfiguration(UpdateStrategy::RightLooking,
                                                   SchedulingStrategy::Level, true); });
        ValidateNumericConfiguration(UpdateStrategy::RightLooking, SchedulingStrategy::Level);
        ValidateNumericConfiguration(UpdateStrategy::LeftLooking, SchedulingStrategy::SynchronizationFree, true);
        std::cout << "Numeric SM padding host configuration checks passed\n";
    } catch (const std::exception &e) {
        std::cerr << e.what() << '\n';
        return 1;
    }
}
