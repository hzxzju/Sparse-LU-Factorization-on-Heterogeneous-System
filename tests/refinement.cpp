#include "refinement.h"
#include <iostream>
#include <stdexcept>

static void check(bool condition, const char *message)
{
    if (!condition) throw std::runtime_error(message);
}

int main()
{
    // Original nonsymmetric CSR system: A=[[4,1],[0,2]], b=[1,1].
    const double values[] = {4, 1, 2};
    const unsigned columns[] = {0, 1, 1}, ptr[] = {0, 2, 3};
    const std::vector<REAL> b = {1, 1};
    unsigned calls = 0;
    // Deliberately inexact triangular solve; every iteration halves the error.
    auto solve = [&](const std::vector<REAL> &r) {
        ++calls;
        const REAL y = r[1] / 2;
        return std::vector<REAL>{(r[0] - y) / 8, y / 2};
    };
    std::vector<REAL> x = {0.125, 0.5};
    auto result = RefineCSR(2, values, columns, ptr, b, x, 1e-6, 2, solve);
    check(result.converged && calls == 0 && result.iterations == 0,
          "already accurate solution must skip correction solves");
    x = {0, 0};
    result = RefineCSR(2, values, columns, ptr, b, x, 1e-6, 30, solve);
    check(result.converged && result.iterations > 1 && result.iterations < 30,
          "inexact factors should converge through repeated corrections");
    check(std::fabs(x[0] - 0.125) < 1e-6 && std::fabs(x[1] - 0.5) < 1e-6,
          "correction sign or CSR orientation is incorrect");
    x = {0, 0};
    result = RefineCSR(2, values, columns, ptr, b, x, 1e-6, 2, solve);
    check(!result.converged && result.iterations == 2 &&
          std::string(result.status) == "max_iterations", "iteration cap must be honored");
    calls = 0;
    x = {0, 0};
    result = RefineCSR(2, values, columns, ptr, b, x, 1e-6, 0, solve);
    check(!result.converged && calls == 0 && result.iterations == 0,
          "zero cap must check residual without corrections");
    result = RefineCSR(2, values, columns, ptr, b, x, 1e-6, 30,
        [](const std::vector<REAL> &) { return std::vector<REAL>{0, 0}; });
    check(!result.converged && std::string(result.status) == "stagnated" && x[0] == 0,
          "stagnation must terminate and preserve the best solution");
    result = RefineCSR(2, values, columns, ptr, b, x, 1e-6, 30,
        [](const std::vector<REAL> &) { return std::vector<REAL>{-1, -1}; });
    check(!result.converged && std::string(result.status) == "stagnated" && x[0] == 0,
          "a worsening correction must be rejected");
    result = RefineCSR(2, values, columns, ptr, b, x, 1e-6, 30,
        [](const std::vector<REAL> &) {
            return std::vector<REAL>{std::numeric_limits<REAL>::quiet_NaN(), 0};
        });
    check(!result.converged && std::string(result.status) == "nonfinite" && x[0] == 0,
          "nonfinite correction must be rejected");
    x[0] = std::numeric_limits<REAL>::infinity();
    calls = 0;
    result = RefineCSR(2, values, columns, ptr, b, x, 1e-6, 30, solve);
    check(!result.converged && calls == 0 && !std::isfinite(result.residual.l2),
          "nonfinite initial residual must not pass");
    // Long-double products retain cancellation information on this platform.
    if (std::numeric_limits<long double>::digits > std::numeric_limits<double>::digits) {
        const double cancel_values[] = {1e16, 1, -1e16};
        const unsigned cancel_columns[] = {0, 1, 2}, cancel_ptr[] = {0, 3, 3, 3};
        std::vector<REAL> r;
        auto norms = ResidualCSR(3, cancel_values, cancel_columns, cancel_ptr,
                                {1, 0, 0}, {1, 1, 1}, r);
        check(norms.l2 == 0, "residual must accumulate in extended precision");
    }
    std::cout << "refinement checks passed\n";
}
