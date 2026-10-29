#include "symbolic.h"
#include <algorithm>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <vector>

static void Require(bool condition, const char *message)
{
    if (!condition) throw std::runtime_error(message);
}

static SymbolicAutoSelection Select(const std::vector<unsigned> &lengths)
{
    std::vector<unsigned> ptr(lengths.size() + 1, 0);
    std::uint64_t total = 0;
    for (size_t j = 0; j < lengths.size(); ++j) {
        Require(lengths[j] <= lengths.size(), "invalid test column length");
        total += lengths[j];
        Require(total <= std::numeric_limits<unsigned>::max(), "test CSC overflow");
        ptr[j + 1] = static_cast<unsigned>(total);
    }
    const std::vector<unsigned> original = ptr;
    const auto result = SelectAutomaticSymbolicBackend(static_cast<unsigned>(lengths.size()), ptr.data());
    Require(ptr == original, "selection modified input pointers");
    return result;
}

int main()
{
    try {
        SymbolicConfig config;
        Require(config.backend == SymbolicBackend::Auto, "default backend");
        Require(ParseSymbolicBackend("auto") == SymbolicBackend::Auto &&
                std::string(SymbolicBackendName(SymbolicBackend::Auto)) == "auto", "auto parser/name");
        config.verify = true; config.fallback_to_cpu = false;
        config.sf_workers = 2; config.pool_entries = config.max_pool_entries = 1;
        ValidateSymbolicConfiguration(config);

        const auto empty = SelectAutomaticSymbolicBackend(0, nullptr);
        Require(empty.backend == SymbolicBackend::CPU && empty.input_nnz == 0 &&
                empty.copy_lower_bound == 0, "zero order");
        Require(Select(std::vector<unsigned>(40000, 0)).backend == SymbolicBackend::CPU,
                "empty columns must select CPU");
        Require(Select(std::vector<unsigned>(40000, 1)).copy_lower_bound == 0,
                "negative predecessor bounds must saturate at zero");

        // The same fully populated column is expensive at the end, while an
        // early lower-triangular column need not collect any predecessors.
        std::vector<unsigned> lengths(40000, 1);
        lengths.front() = 40000;
        Require(Select(lengths).backend == SymbolicBackend::CPU, "early long column");
        lengths.front() = 1; lengths.back() = 40000;
        Require(Select(lengths).backend == SymbolicBackend::GPU_SF, "late long column");

        // Hold the other three conditions true on both sides of n=20000.
        for (unsigned n : {19999u, 20000u}) {
            lengths.assign(n, 1);
            std::fill(lengths.end() - 3, lengths.end(), n);
            const auto result = Select(lengths);
            Require(result.copy_lower_bound >= UINT64_C(1000000000), "order fixture cost");
            Require(result.backend == (n == 20000 ? SymbolicBackend::GPU_SF : SymbolicBackend::CPU),
                    "order threshold boundary");
        }

        // An exact 10% column density is accepted, one entry less is rejected.
        for (unsigned maximum : {9999u, 10000u}) {
            lengths.assign(100000, 0);
            std::fill(lengths.end() - 11, lengths.end(), maximum);
            const auto result = Select(lengths);
            Require(result.copy_lower_bound >= UINT64_C(1000000000), "density fixture cost");
            Require(result.backend == (maximum == 10000 ? SymbolicBackend::GPU_SF : SymbolicBackend::CPU),
                    "density threshold boundary");
        }

        // Make max/mean exactly 64, then increase nnz by one. Fractional means
        // must not be truncated, and no floating-point tolerance is applied.
        lengths.assign(65536, 0);
        std::fill(lengths.end() - 20, lengths.end(), 8192);
        const std::uint64_t target_nnz = UINT64_C(8192) * 65536 / 64;
        std::uint64_t remaining = target_nnz - 20 * 8192;
        for (size_t j = 0; remaining; ++j) {
            lengths[j] = static_cast<unsigned>(std::min<std::uint64_t>(remaining, 8191));
            remaining -= lengths[j];
        }
        Require(Select(lengths).backend == SymbolicBackend::GPU_SF, "imbalance equality");
        ++lengths[2000];
        Require(Select(lengths).backend == SymbolicBackend::CPU, "fractional mean below imbalance threshold");

        // Ten late 10,000-entry columns contribute 999,450,000. A column at
        // n-j=450 with 1,000 entries adds exactly 550,000 to reach 10^9.
        lengths.assign(100000, 0);
        std::fill(lengths.end() - 10, lengths.end(), 10000);
        lengths[100000 - 450] = 1000;
        const auto exact_cost = Select(lengths);
        Require(exact_cost.copy_lower_bound == UINT64_C(1000000000) &&
                exact_cost.backend == SymbolicBackend::GPU_SF, "cost equality");
        --lengths[100000 - 450];
        Require(Select(lengths).backend == SymbolicBackend::CPU, "cost below threshold");

        lengths.assign(100000, 0); lengths.back() = 100000;
        const auto large_cost = Select(lengths);
        Require(large_cost.copy_lower_bound == UINT64_C(9999900000) &&
                large_cost.backend == SymbolicBackend::GPU_SF, "64-bit cost accumulation");
        std::cout << "PASS automatic symbolic policy: defaults, all threshold boundaries, "
                     "column positions, empty input and 64-bit accumulation\n";
    } catch (const std::exception &error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
