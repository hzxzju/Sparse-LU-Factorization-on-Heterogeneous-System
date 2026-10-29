#ifndef GLU_REFINEMENT_H
#define GLU_REFINEMENT_H

#include "type.h"
#include <algorithm>
#include <cmath>
#include <limits>
#include <ostream>
#include <vector>

struct ResidualNorms {
    long double l1 = 0, l2 = 0, linf = 0;
};

// Always evaluate the original, unscaled CSR system. The solution may retain
// extended precision while residual right-hand sides and solves stay REAL.
template <class SolutionScalar = REAL>
inline ResidualNorms ResidualCSR(unsigned n, const double *values,
        const unsigned *columns, const unsigned *ptr,
        const std::vector<REAL> &b, const std::vector<SolutionScalar> &x,
        std::vector<REAL> &residual)
{
    ResidualNorms norms;
    long double scale = 0, squares = 1;
    residual.resize(n);
    for (unsigned i = 0; i < n; ++i) {
        long double sum = 0;
        for (unsigned p = ptr[i]; p < ptr[i + 1]; ++p)
            sum += static_cast<long double>(values[p]) * x[columns[p]];
        const long double r = static_cast<long double>(b[i]) - sum;
        residual[i] = static_cast<REAL>(r);
        if (!std::isfinite(r) || !std::isfinite(residual[i])) {
            norms.l1 = norms.l2 = norms.linf = std::numeric_limits<long double>::infinity();
            return norms;
        }
        const long double magnitude = std::fabs(r);
        norms.l1 += magnitude;
        norms.linf = std::max(norms.linf, magnitude);
        // Scaled sum of squares avoids overflow/underflow in the L2 norm.
        if (magnitude != 0) {
            if (scale < magnitude) {
                const long double ratio = scale / magnitude;
                squares = 1 + squares * ratio * ratio;
                scale = magnitude;
            } else {
                const long double ratio = magnitude / scale;
                squares += ratio * ratio;
            }
        }
    }
    norms.l2 = scale * std::sqrt(squares);
    return norms;
}

struct RefinementResult {
    ResidualNorms residual;
    long double initial_l2 = 0;
    unsigned iterations = 0;  // attempted correction solves
    const char *status = "max_iterations";
    bool converged = false;
};

// solve(rhs) must reuse the existing factors, including their permutations and
// scales. Keep the best finite solution; stop on stagnation or invalid arithmetic
// rather than spending up to n iterations on a correction that cannot help.
template <class SolutionScalar, class Solve>
RefinementResult RefineCSR(unsigned n, const double *values,
        const unsigned *columns, const unsigned *ptr,
        const std::vector<REAL> &b, std::vector<SolutionScalar> &x,
        long double tolerance, unsigned max_iterations, Solve solve,
        std::ostream *progress = nullptr)
{
    RefinementResult result;
    std::vector<REAL> residual, candidate_residual;
    result.residual = ResidualCSR(n, values, columns, ptr, b, x, residual);
    result.initial_l2 = result.residual.l2;
    if (progress)
        *progress << "Refinement initial residual L2: " << result.initial_l2 << std::endl;
    if (!std::isfinite(result.residual.l2)) {
        result.status = "nonfinite";
        return result;
    }
    while (result.residual.l2 > tolerance && result.iterations < max_iterations) {
        std::vector<REAL> correction = solve(residual);
        ++result.iterations;
        std::vector<SolutionScalar> candidate(n);
        bool finite = correction.size() == n;
        if (finite) {
            for (unsigned i = 0; i < n; ++i) {
                candidate[i] = x[i] + static_cast<SolutionScalar>(correction[i]);
                finite = finite && std::isfinite(candidate[i]);
            }
        }
        if (!finite) {
            result.status = "nonfinite";
            return result;
        }
        const ResidualNorms norms = ResidualCSR(n, values, columns, ptr, b,
                                               candidate, candidate_residual);
        if (progress)
            *progress << "Refinement iteration " << result.iterations
                      << " residual L2: " << norms.l2 << std::endl;
        if (!std::isfinite(norms.l2)) {
            result.status = "nonfinite";
            return result;
        }
        if (norms.l2 >= result.residual.l2) {
            result.status = "stagnated";
            return result;
        }
        x.swap(candidate);
        residual.swap(candidate_residual);
        result.residual = norms;
    }
    result.converged = result.residual.l2 <= tolerance;
    result.status = result.converged ? "converged" : "max_iterations";
    return result;
}

#endif
