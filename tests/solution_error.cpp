#include "solution_error.h"
#include <iostream>
#include <sstream>

static void check(bool condition, const char *message)
{
    if (!condition) throw std::runtime_error(message);
}

int main()
{
    std::istringstream input("0.125\n0.5\n");
    const auto cpu = ReadCPUReference(input, 2);
    check(cpu[0] == 0.125 && cpu[1] == 0.5, "wrong CPU reference values");
    auto error = CompareSolutions(std::vector<long double>{0.125L, 0.5L}, cpu);
    check(error.absolute_l2 == 0 && error.relative_l2 == 0, "equal solutions");
    error = CompareSolutions(std::vector<double>{3, 4}, {0, 0});
    check(error.absolute_l2 == 5 && std::isinf(error.relative_l2), "zero reference");
    error = CompareSolutions(std::vector<double>{0, 0}, {0, 0});
    check(error.relative_l2 == 0, "both zero solutions");
    error = CompareSolutions(std::vector<double>{6, 8}, {3, 4});
    check(error.absolute_l2 == 5 && error.relative_l2 == 1, "absolute/relative definition");
    error = CompareSolutions(std::vector<double>{1e300, 1e300}, {0, 0});
    check(std::isfinite(error.absolute_l2) && error.absolute_l2 > 1e300, "large norm overflow");
    error = CompareSolutions(std::vector<double>{1e-300, 1e-300}, {0, 0});
    check(error.absolute_l2 > 1e-300, "small norm underflow");
    if (std::numeric_limits<long double>::digits > std::numeric_limits<double>::digits) {
        error = CompareSolutions(std::vector<long double>{1.L + 1e-18L}, {1.});
        check(error.absolute_l2 > 0, "extended precision solution must not be rounded");
    }
    bool rejected = false;
    try { CompareSolutions(std::vector<double>{1}, {1, 2}); }
    catch (const std::invalid_argument &) { rejected = true; }
    check(rejected, "mismatched solution sizes accepted");
    rejected = false;
    try { CompareSolutions(std::vector<double>{NAN}, {1}); }
    catch (const std::runtime_error &) { rejected = true; }
    check(rejected, "nonfinite solution accepted");
    for (const char *contents : {"1\n", "1\n2\n3\n", "1\nnan\n", "1\ninf\n", "1\n2\njunk"}) {
        rejected = false;
        std::istringstream invalid(contents);
        try { ReadCPUReference(invalid, 2); }
        catch (const std::runtime_error &) { rejected = true; }
        check(rejected, "malformed CPU reference accepted");
    }
    rejected = false;
    try { ReadCPUReference("/tmp/glu-no-such-reference/absent.txt", 2); }
    catch (const std::runtime_error &) { rejected = true; }
    check(rejected, "missing reference file accepted");
    std::cout << "solution comparison checks passed\n";
}
