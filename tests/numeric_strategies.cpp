#include "numeric.h"
#include <algorithm>
#include <cmath>
#include <random>
#include <sstream>
#include <stdexcept>
#include <vector>

// Exercise the production symbolic phase, strategy dispatcher and GPU kernels.
// The dense CPU reference deliberately uses an independent elimination order.
static std::vector<REAL> dense_reference(std::vector<REAL> a, unsigned n, bool perturb)
{
    REAL norm = 0;
    for (unsigned j = 0; j < n; ++j) {
        REAL sum = 0;
        for (unsigned i = 0; i < n; ++i) sum += std::fabs(a[i*n+j]);
        norm = std::max(norm, sum);
    }
    const REAL threshold = 3.45e-4 * norm;
    for (unsigned k = 0; k < n; ++k) {
        if (perturb && std::fabs(a[k*n+k]) < threshold) a[k*n+k] = threshold;
        for (unsigned i = k + 1; i < n; ++i) {
            a[i*n+k] /= a[k*n+k];
            if (a[i*n+k] == 0) continue;
            for (unsigned j = k + 1; j < n; ++j)
                a[i*n+j] -= a[i*n+k] * a[k*n+j];
        }
    }
    return a;
}

static REAL run_case(const char *name, const std::vector<REAL> &a, unsigned n,
                     bool perturb = false)
{
    std::vector<unsigned> rows, ptr(1, 0);
    std::vector<double> values;
    for (unsigned j = 0; j < n; ++j) {
        for (unsigned i = 0; i < n; ++i) if (a[i*n+j] != 0) {
            rows.push_back(i);
            values.push_back(a[i*n+j]);
        }
        ptr.push_back(static_cast<unsigned>(rows.size()));
    }
    const std::vector<REAL> reference = dense_reference(a, n, perturb);
    std::vector<REAL> previous;
    REAL worst = 0;
    struct Mode { UpdateStrategy strategy; SchedulingStrategy scheduling; unsigned workers; };
    const Mode modes[] = {
        {UpdateStrategy::RightLooking, SchedulingStrategy::Level, 0},
        {UpdateStrategy::LeftLooking, SchedulingStrategy::Level, 0},
        {UpdateStrategy::LeftLooking, SchedulingStrategy::SynchronizationFree, 1},
        {UpdateStrategy::LeftLooking, SchedulingStrategy::SynchronizationFree, 0},
        {UpdateStrategy::LeftLooking, SchedulingStrategy::SynchronizationFree, 3}
    };
    for (const Mode &mode : modes) {
        const UpdateStrategy strategy = mode.strategy;
        std::ostringstream out, err;
        Symbolic_Matrix sym(n, out, err);
        sym.fill_in(rows.data(), ptr.data());
        // SF must work without a CPU level graph or GPU CSR data.
        if (mode.scheduling == SchedulingStrategy::Level)
            sym.csr();
        sym.predictLU(rows.data(), ptr.data(), values.data());
        if (mode.scheduling == SchedulingStrategy::Level)
            sym.leveling();
        LUonDevice(sym, out, err, perturb, strategy, mode.scheduling, mode.workers);
        if (mode.scheduling == SchedulingStrategy::SynchronizationFree) {
            // Repeat on the same symbolic structure and different input values.
            // CPU symbolic structure is reusable, device done/counter are not.
            const std::vector<REAL> first_factors = sym.val;
            for (unsigned repeat = 0; repeat < 2; ++repeat) {
                sym.val.clear();
                std::vector<double> scaled = values;
                const REAL scale = repeat == 0 ? 1.5 : 0.75;
                for (double &value : scaled) value *= scale;
                sym.predictLU(rows.data(), ptr.data(), scaled.data());
                LUonDevice(sym, out, err, perturb, strategy, mode.scheduling, mode.workers);
                for (unsigned k = 0; k < n; ++k) {
                    for (unsigned p = sym.sym_c_ptr[k]; p < sym.sym_c_ptr[k + 1]; ++p) {
                        const REAL expected = first_factors[p] *
                            (sym.sym_r_idx[p] <= k ? scale : 1.0);
                        if (!std::isfinite(sym.val[p]) ||
                            std::fabs(sym.val[p] - expected) > 1e-10 * (1 + std::fabs(expected)))
                            throw std::runtime_error(std::string(name) + ": repeated SF factorization");
                    }
                }
            }
            sym.val = first_factors;
        }
        if (!err.str().empty()) throw std::runtime_error(err.str());
        for (unsigned j = 0; j < n; ++j) {
            for (unsigned p = sym.sym_c_ptr[j]; p < sym.sym_c_ptr[j+1]; ++p) {
                const REAL expected = reference[sym.sym_r_idx[p]*n+j];
                const REAL delta = std::fabs(sym.val[p] - expected) / (1 + std::fabs(expected));
                if (!std::isfinite(sym.val[p]) || delta > 1e-10) {
                    std::ostringstream detail;
                    detail << name << ": " << UpdateStrategyName(strategy)
                           << " factor (" << sym.sym_r_idx[p] << ',' << j << ")="
                           << sym.val[p] << ", CPU=" << expected;
                    throw std::runtime_error(detail.str());
                }
                worst = std::max(worst, delta);
            }
        }
        if (!previous.empty()) for (unsigned p = 0; p < sym.nnz; ++p) {
            if (std::fabs(sym.val[p]-previous[p]) > 1e-10*(1+std::fabs(previous[p])))
                throw std::runtime_error(std::string(name) + ": RL/LL factor mismatch");
        }
        previous = sym.val;

        if (!perturb) {
            // Reconstruct L*U using sparse factors, including L's unit diagonal.
            std::vector<std::vector<std::pair<unsigned, REAL>>> urows(n);
            for (unsigned j = 0; j < n; ++j)
                for (unsigned p = sym.sym_c_ptr[j]; p <= sym.l_col_ptr[j]; ++p)
                    urows[sym.sym_r_idx[p]].push_back({j, sym.val[p]});
            std::vector<REAL> product(n*n, 0);
            for (unsigned k = 0; k < n; ++k) {
                for (const auto &u : urows[k]) product[k*n+u.first] += u.second;
                for (unsigned p = sym.l_col_ptr[k]+1; p < sym.sym_c_ptr[k+1]; ++p)
                    for (const auto &u : urows[k])
                        product[sym.sym_r_idx[p]*n+u.first] += sym.val[p]*u.second;
            }
            for (unsigned p = 0; p < n*n; ++p) {
                const REAL delta = std::fabs(product[p]-a[p]) / (1+std::fabs(a[p]));
                if (delta > 1e-10) throw std::runtime_error(std::string(name) + ": L*U != A");
                worst = std::max(worst, delta);
            }

            // Known solution, followed by forward/backward substitution and Ax-b.
            std::vector<REAL> b(n, 0), x;
            for (unsigned i = 0; i < n; ++i)
                for (unsigned j = 0; j < n; ++j) b[i] += a[i*n+j]*(1 + (j%7)*0.125);
            x = b;
            for (unsigned k = 0; k < n; ++k)
                for (unsigned p = sym.l_col_ptr[k]+1; p < sym.sym_c_ptr[k+1]; ++p)
                    x[sym.sym_r_idx[p]] -= sym.val[p]*x[k];
            for (unsigned kk = n; kk > 0; --kk) {
                const unsigned k = kk-1;
                x[k] /= sym.val[sym.l_col_ptr[k]];
                for (unsigned p = sym.sym_c_ptr[k]; p < sym.l_col_ptr[k]; ++p)
                    x[sym.sym_r_idx[p]] -= sym.val[p]*x[k];
            }
            for (unsigned i = 0; i < n; ++i) {
                REAL sum = 0;
                for (unsigned j = 0; j < n; ++j) sum += a[i*n+j]*x[j];
                const REAL delta = std::fabs(sum-b[i])/(1+std::fabs(b[i]));
                if (!std::isfinite(delta) || delta > 1e-10)
                    throw std::runtime_error(std::string(name) + ": Ax-b residual");
                worst = std::max(worst, delta);
            }
        }
    }
    std::cout << "PASS " << name << ": max relative error " << worst << '\n';
    return worst;
}

static std::vector<REAL> independent_blocks(unsigned blocks)
{
    const unsigned n = 2*blocks;
    std::vector<REAL> a(n*n, 0);
    for (unsigned b = 0; b < blocks; ++b) {
        const unsigned k = 2*b;
        a[k*n+k] = 4;
        a[k*n+k+1] = 0.5;
        a[(k+1)*n+k] = 1;
        a[(k+1)*n+k+1] = 3;
    }
    return a;
}

int main()
{
    try {
        for (const char *alias : {"left-looking", "left", "ll"})
            if (ParseUpdateStrategy(alias) != UpdateStrategy::LeftLooking) return 1;
        for (const char *alias : {"right-looking", "right", "rl"})
            if (ParseUpdateStrategy(alias) != UpdateStrategy::RightLooking) return 1;
        bool rejected = false;
        try { ParseUpdateStrategy("invalid"); } catch (const std::invalid_argument &) { rejected = true; }
        if (!rejected) return 1;
        if (ParseSchedulingStrategy("level") != SchedulingStrategy::Level ||
            ParseSchedulingStrategy("synchronization-free") != SchedulingStrategy::SynchronizationFree)
            return 1;
        rejected = false;
        try { ParseSchedulingStrategy("invalid"); } catch (const std::invalid_argument &) { rejected = true; }
        if (!rejected) return 1;
        rejected = false;
        try { ValidateNumericConfiguration(UpdateStrategy::RightLooking,
                                           SchedulingStrategy::SynchronizationFree); }
        catch (const std::invalid_argument &) { rejected = true; }
        if (!rejected) return 1;
        {
            std::ostringstream out, err;
            Symbolic_Matrix invalid(2, out, err);
            invalid.nnz = 2;
            invalid.sym_c_ptr = {0, 1, 2};
            invalid.sym_r_idx = {1, 0}; // invalid diagonal / forward dependency
            invalid.l_col_ptr = {0, 1};
            invalid.val = {2, 3};
            LUonDevice(invalid, out, err, false, UpdateStrategy::LeftLooking,
                       SchedulingStrategy::SynchronizationFree);
            if (err.str().empty()) return 1;
        }
        run_case("scalar", {2}, 1);
        run_case("upper-empty-L", {2,1,2,0,3,1,0,0,4}, 3);
        run_case("missing-structural-diagonal", {2,1,0,1,0,1,0,1,2}, 3);
        run_case("fill-in", {4,0,2,0,1,3,0,1,0,2,5,0,1,0,1,4}, 4);
        run_case("shared-successor", {4,0,1,2,0,5,2,1,1,2,6,1,2,1,1,7}, 4);
        run_case("zero-pivot-perturb", {1,1,2,1,1,2,2,2,5}, 3, true);
        run_case("negative-small-pivot-perturb", {1,1,2,1,1-1e-12,2,2,2,5}, 3, true);
        for (unsigned blocks : {1u,16u,17u,449u,897u}) {
            const std::string name = "parallel-blocks-" + std::to_string(blocks);
            run_case(name.c_str(), independent_blocks(blocks), 2*blocks);
        }
        // First level has one column; the wider second level forces workspace
        // reuse with a batch size of one and differently shaped target columns.
        std::vector<REAL> arrow(41*41, 0);
        for (unsigned i = 0; i < 41; ++i) {
            arrow[i*41+i] = 4;
            if (i > 0) arrow[i*41] = 0.25;
        }
        run_case("wider-second-level", arrow, 41);
        std::mt19937 rng(20261005);
        std::uniform_real_distribution<REAL> dist(-0.5, 0.5);
        for (unsigned n : {7u,31u,65u,260u}) {
            std::vector<REAL> a(n*n, 0);
            for (unsigned i = 0; i < n; ++i) {
                REAL row_sum = 0;
                for (unsigned j = 0; j < n; ++j) if (i != j && (n == 260 || rng()%5 == 0)) {
                    a[i*n+j] = dist(rng);
                    row_sum += std::fabs(a[i*n+j]);
                }
                a[i*n+i] = 1 + row_sum;
            }
            const std::string name = "random-diagonally-dominant-" + std::to_string(n);
            run_case(name.c_str(), a, n);
        }
        std::cout << "All 17 numeric cases passed (RL-level, LL-level and three LL-SF worker counts).\n";
    } catch (const std::exception &e) {
        std::cerr << e.what() << '\n';
        return 1;
    }
}
