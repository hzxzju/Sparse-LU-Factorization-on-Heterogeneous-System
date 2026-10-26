#include <hc_runtime.h>
#include "symbolic.h"
#if defined(GLU_SYMBOLIC_TEST_NUMERIC)
#include "numeric.h"
#endif

#include <algorithm>
#include <cmath>
#include <functional>
#include <iostream>
#include <initializer_list>
#include <limits>
#include <random>
#include <sstream>
#include <stdexcept>

namespace {
SymbolicBackend test_backend = SymbolicBackend::GPU;


void Require(bool condition, const std::string &message)
{
    if (!condition) throw std::runtime_error(message);
}

void Reject(const std::function<void()> &operation)
{
    bool rejected = false;
    try { operation(); } catch (const std::invalid_argument &) { rejected = true; }
    Require(rejected, "invalid symbolic input/configuration was accepted");
}

struct Matrix
{
    unsigned n;
    std::vector<unsigned> rows, ptr;
    std::vector<double> values;
};

template <typename T>
void Equal(const std::vector<T> &cpu, const std::vector<T> &gpu, const char *name)
{
    Require(cpu.size() == gpu.size(), std::string(name) + ": size mismatch");
    for (size_t i = 0; i < cpu.size(); ++i) {
        if (cpu[i] == gpu[i]) continue;
        std::ostringstream message;
        message << name << '[' << i << "]: CPU=" << cpu[i] << ", GPU=" << gpu[i];
        throw std::runtime_error(message.str());
    }
}

Matrix RandomMatrix(unsigned n, unsigned seed, bool dense = false)
{
    Matrix a{n, {}, {0}, {}};
    std::mt19937 rng(seed);
    for (unsigned j = 0; j < n; ++j) {
        std::vector<unsigned> rows;
        for (unsigned r = 0; r < n; ++r)
            if (dense || rng() % 7 == 0) rows.push_back(r);
        std::shuffle(rows.begin(), rows.end(), rng);
        for (unsigned r : rows) {
            a.rows.push_back(r);
            a.values.push_back(1.0 + r + j * 0.125);
        }
        a.ptr.push_back(static_cast<unsigned>(a.rows.size()));
    }
    return a;
}

Matrix StructuredMatrix(unsigned n, unsigned kind)
{
    Matrix a{n, {}, {0}, {}};
    for (unsigned col = 0; col < n; ++col) {
        for (unsigned row = 0; row < n; ++row) {
            bool present = kind == 0 ? row == col :
                kind == 1 ? row <= col : kind == 2 ? row >= col :
                kind == 3 ? (row + 2 >= col && col + 2 >= row) :
                kind == 4 ? row / 8 == col / 8 : (row == 0 || row == col);
            if (present) { a.rows.push_back(row); a.values.push_back(row == col ? 1000. : .125); }
        }
        a.ptr.push_back(static_cast<unsigned>(a.rows.size()));
    }
    return a;
}

void Prepare(Symbolic_Matrix &sym, Matrix &a)
{
    sym.csr();
    sym.predictLU(a.rows.data(), a.ptr.data(), a.values.data());
    sym.leveling();
}

void Compare(const Symbolic_Matrix &cpu, const Symbolic_Matrix &gpu)
{
    Require(cpu.nnz == gpu.nnz, "nnz mismatch");
    Equal(cpu.sym_c_ptr, gpu.sym_c_ptr, "sym_c_ptr");
    Equal(cpu.sym_r_idx, gpu.sym_r_idx, "sym_r_idx");
    Equal(cpu.l_col_ptr, gpu.l_col_ptr, "l_col_ptr");
    // These arrays are constructed on CPU for both patterns in Phase 1.
    Equal(cpu.csr_r_ptr, gpu.csr_r_ptr, "csr_r_ptr");
    Equal(cpu.csr_c_idx, gpu.csr_c_idx, "csr_c_idx");
    Equal(cpu.csr_diag_ptr, gpu.csr_diag_ptr, "csr_diag_ptr");
    Equal(cpu.val, gpu.val, "initial val");
    Require(cpu.num_lev == gpu.num_lev, "num_lev mismatch");
    Equal(cpu.level_ptr, gpu.level_ptr, "level_ptr");
    Equal(cpu.level_idx, gpu.level_idx, "level_idx");
    for (unsigned j = 0; j < gpu.n; ++j) {
        Require(gpu.sym_c_ptr[j] <= gpu.l_col_ptr[j] &&
                gpu.l_col_ptr[j] < gpu.sym_c_ptr[j + 1], "CSC diagonal outside column");
        Require(gpu.csr_r_ptr[j] <= gpu.csr_diag_ptr[j] &&
                gpu.csr_diag_ptr[j] < gpu.csr_r_ptr[j + 1], "CSR diagonal outside row");
        Require(gpu.sym_r_idx[gpu.l_col_ptr[j]] == j, "non-absolute CSC diagonal");
        Require(gpu.csr_c_idx[gpu.csr_diag_ptr[j]] == j, "non-absolute CSR diagonal");
        for (unsigned p = gpu.sym_c_ptr[j] + 1; p < gpu.sym_c_ptr[j + 1]; ++p)
            Require(gpu.sym_r_idx[p - 1] < gpu.sym_r_idx[p], "unsorted/duplicate CSC rows");
        for (unsigned p = gpu.csr_r_ptr[j] + 1; p < gpu.csr_r_ptr[j + 1]; ++p)
            Require(gpu.csr_c_idx[p - 1] < gpu.csr_c_idx[p], "unsorted/duplicate CSR columns");
    }
}

// Check the design's path criterion independently of the GPU queue algorithm.
// All-pairs bottleneck Floyd-Warshall includes every internal vertex, then
// filters by min(row,column). This is host mathematics validation only.
void GraphSemanticsChecks()
{
    unsigned cases = 0;
    std::ostringstream out, err;
    for (unsigned n = 1; n <= 4; ++n) {
        Symbolic_Matrix cpu(n, out, err);
        for (unsigned mask = 0; mask < (1u << (n * n)); ++mask) {
            Matrix a{n, {}, {0}, {}};
            std::vector<unsigned> best(n * n, std::numeric_limits<unsigned>::max());
            for (unsigned j = 0; j < n; ++j) {
                best[j * n + j] = 0;
                for (unsigned r = n; r-- > 0;) {
                    if (!(mask & (1u << (j * n + r)))) continue;
                    a.rows.push_back(r); // deliberately unsorted
                    best[j * n + r] = 0;
                }
                a.ptr.push_back(static_cast<unsigned>(a.rows.size()));
            }
            for (unsigned k = 0; k < n; ++k)
                for (unsigned j = 0; j < n; ++j)
                    for (unsigned r = 0; r < n; ++r)
                        best[j * n + r] = std::min(best[j * n + r],
                            std::max(k + 1, std::max(best[j * n + k], best[k * n + r])));
            std::vector<unsigned> ptr(1, 0), rows, diagonal;
            for (unsigned j = 0; j < n; ++j) {
                for (unsigned r = 0; r < n; ++r) {
                    if (r != j && best[j * n + r] > std::min(r, j)) continue;
                    if (r == j) diagonal.push_back(static_cast<unsigned>(rows.size()));
                    rows.push_back(r);
                }
                ptr.push_back(static_cast<unsigned>(rows.size()));
            }
            out.str("");
            cpu.fill_in(a.rows.data(), a.ptr.data());
            Equal(cpu.sym_c_ptr, ptr, "host graph sym_c_ptr");
            Equal(cpu.sym_r_idx, rows, "host graph sym_r_idx");
            Equal(cpu.l_col_ptr, diagonal, "host graph l_col_ptr");
            ++cases;
        }
    }
    std::cout << "PASS host path criterion vs CPU fill_in: " << cases
              << " exhaustive patterns (not GPU execution)\n";
}

void LevelSemanticsChecks()
{
    const Matrix matrices[] = {
        {2, {0}, {0, 0, 1}, {1}},                 // U predecessor has no strict L
        {2, {1}, {0, 1, 1}, {1}},                 // L rule alone
        {3, {2, 0}, {0, 1, 2, 2}, {1, 1}},       // U rule, then L predecessors
        {2, {1, 0}, {0, 1, 2}, {1, 1}}           // same edge from both rules
    };
    const std::vector<unsigned> levels[] = {{0, 0}, {0, 1}, {0, 1, 2}, {0, 1}};
    for (unsigned test = 0; test < 4; ++test) {
        Matrix a = matrices[test];
        std::ostringstream out, err;
        Symbolic_Matrix cpu(a.n, out, err);
        cpu.fill_in(a.rows.data(), a.ptr.data());
        Prepare(cpu, a);
        std::vector<unsigned> actual(a.n);
        for (int lev = 0; lev < cpu.num_lev; ++lev)
            for (int p = cpu.level_ptr[lev]; p < cpu.level_ptr[lev + 1]; ++p)
                actual[cpu.level_idx[p]] = static_cast<unsigned>(lev);
        Equal(levels[test], actual, "CPU leveling semantics");
        Compare(cpu, cpu); // also check sorted rows and absolute diagonals
    }
    std::cout << "PASS CPU U/L leveling rules, CSR order and absolute diagonals\n";
}

void HostChecks()
{
    Require(ParseSymbolicBackend("cpu") == SymbolicBackend::CPU &&
            ParseSymbolicBackend("gpu") == SymbolicBackend::GPU &&
            ParseSymbolicBackend("gpu-sf") == SymbolicBackend::GPU_SF, "backend parser");
    for (SymbolicBackend backend : {SymbolicBackend::CPU, SymbolicBackend::GPU, SymbolicBackend::GPU_SF})
        Require(ParseSymbolicBackend(SymbolicBackendName(backend)) == backend, "backend name roundtrip");
    Reject([] { SymbolicBackendName(static_cast<SymbolicBackend>(99)); });
    Reject([] { ParseSymbolicBackend("bad"); });
    SymbolicConfig config;
    ValidateSymbolicConfiguration(config);
    config.batch_cols = 0;
    Reject([&] { ValidateSymbolicConfiguration(config); });
    config.batch_cols = 1; config.block_size = 96;
    Reject([&] { ValidateSymbolicConfiguration(config); });
    config.block_size = 256; config.verify = true;
    Reject([&] { ValidateSymbolicConfiguration(config); });
    config.verify = false; config.fallback_to_cpu = false;
    Reject([&] { ValidateSymbolicConfiguration(config); });
    config.backend = SymbolicBackend::GPU_SF;
    config.sf_workers = 3; config.pool_entries = 2; config.max_pool_entries = 1;
    Reject([&] { ValidateSymbolicConfiguration(config); });
    config.max_pool_entries = 2;
    ValidateSymbolicConfiguration(config);
    config.sf_workers = config.pool_entries = config.max_pool_entries = 0;
    config.backend = test_backend;
    std::ostringstream out, err;
    Symbolic_Matrix invalid(2, out, err);
    unsigned rows[] = {0, 0}, ptr[] = {0, 2, 2};
    Reject([&] { invalid.fill_in(rows, ptr, config); }); // duplicate
    SymbolicConfig cpu_config;
    Reject([&] { invalid.fill_in(rows, ptr, cpu_config); }); // same configured CPU contract
    rows[1] = 2;
    Reject([&] { invalid.fill_in(rows, ptr, config); }); // out of range
    ptr[1] = 3;
    Reject([&] { invalid.fill_in(rows, ptr, config); }); // nonmonotone
    ptr[1] = 0; ptr[0] = 1;
    Reject([&] { invalid.fill_in(rows, ptr, config); }); // nonzero base
    ptr[0] = 0; ptr[1] = 1;
    Reject([&] { invalid.fill_in(nullptr, ptr, config); });
    Reject([&] { invalid.fill_in(rows, nullptr, config); });

    Matrix a = RandomMatrix(16, 17);
    Symbolic_Matrix cpu(a.n, out, err), repeated(a.n, out, err);
    cpu.fill_in(a.rows.data(), a.ptr.data());
    Prepare(cpu, a);
    config = SymbolicConfig();
    for (unsigned repeat = 0; repeat < 3; ++repeat) {
        repeated.fill_in(a.rows.data(), a.ptr.data(), config);
        Require(repeated.symbolic_backend == SymbolicBackend::CPU, "CPU backend identity");
        Prepare(repeated, a);
        repeated.csr(); repeated.predictLU(a.rows.data(), a.ptr.data(), a.values.data());
        repeated.leveling();
        Compare(cpu, repeated);
    }
    // Empty original columns still receive one structural diagonal, zero value.
    Matrix empty{2, {}, {0, 0, 0}, {}};
    repeated.n = 2;
    repeated.fill_in(nullptr, empty.ptr.data());
    Prepare(repeated, empty);
    Require(repeated.sym_r_idx == std::vector<unsigned>({0, 1}) &&
            repeated.val == std::vector<REAL>({0, 0}), "empty original columns");
    unsigned zero = 0;
    Symbolic_Matrix zero_order(0, out, err);
    zero_order.csr_r_ptr = {5}; zero_order.csr_c_idx = {5}; zero_order.csr_diag_ptr = {5};
    zero_order.val = {9}; zero_order.level_ptr = {5}; zero_order.level_idx = {5}; zero_order.num_lev = 5;
#if GLU_DEBUG
    zero_order.CCA = {9};
#endif
    config.backend = test_backend; config.verify = true;
    config.fallback_to_cpu = false;
    zero_order.fill_in(nullptr, &zero, config);
    if (test_backend == SymbolicBackend::GPU_SF)
        Require(out.str().find("symbolic_gpu_compute_ms: 0") != std::string::npos &&
                out.str().find("symbolic_workers: 0") != std::string::npos,
                "empty SF result omitted zero resource/compute metrics");
    Require(zero_order.symbolic_backend == test_backend, "empty GPU backend identity");
    Require(zero_order.csr_r_ptr.empty() && zero_order.csr_c_idx.empty() && zero_order.csr_diag_ptr.empty() &&
            zero_order.val.empty() && zero_order.level_ptr.empty() && zero_order.level_idx.empty() &&
            zero_order.num_lev == 0, "GPU commit did not invalidate all caches");
#if GLU_DEBUG
    Require(zero_order.CCA.empty(), "GPU commit did not invalidate debug cache");
#endif
    zero_order.csr(); zero_order.leveling();
    Require(zero_order.nnz == 0 && zero_order.num_lev == 0 &&
            zero_order.sym_c_ptr == std::vector<unsigned>({0}), "zero order");
    std::cout << "PASS host configuration, validation, empty inputs and repeated analysis\n";
}

void PatternCase(const char *name, Matrix a,
                 std::initializer_list<unsigned> threads_list = {64u, 128u, 256u},
                 std::initializer_list<unsigned> batch_list = {1u, 3u, 128u})
{
    std::cout << "RUN " << name << " n=" << a.n << std::endl;
    const std::vector<unsigned> original_rows = a.rows, original_ptr = a.ptr;
    const std::vector<double> original_values = a.values;
    std::ostringstream out, err;
    Symbolic_Matrix cpu(a.n, out, err), gpu(a.n, out, err);
    cpu.fill_in(a.rows.data(), a.ptr.data());
    Prepare(cpu, a);
    // Reuse the same object with different slot counts and CTA sizes.
    for (unsigned threads : threads_list) {
        for (unsigned batch : batch_list) {
            SymbolicConfig config;
            config.backend = test_backend;
            config.batch_cols = batch; config.block_size = threads;
            if (test_backend == SymbolicBackend::GPU_SF) config.sf_workers = batch;
            config.verify = true; config.fallback_to_cpu = false;
            gpu.fill_in(a.rows.data(), a.ptr.data(), config);
            Require(gpu.symbolic_backend == test_backend, "GPU backend identity");
            Require(gpu.gpu_device >= 0, "successful GPU analysis did not retain its selected device");
            Equal(original_rows, a.rows, "input rows changed");
            Equal(original_ptr, a.ptr, "input pointers changed");
            Equal(original_values, a.values, "input values changed");
            Require(gpu.val.empty() && gpu.csr_r_ptr.empty() && gpu.level_ptr.empty(),
                    "reanalysis kept derived arrays");
            Prepare(gpu, a);
            Compare(cpu, gpu);
        }
    }
    std::cout << "PASS " << name << " n=" << a.n << std::endl;
}

#if defined(GLU_SYMBOLIC_TEST_NUMERIC)
void NumericChecks()
{
    Matrix a{4, {0, 1, 3, 0, 1, 2, 1, 2, 3, 0, 2, 3},
             {0, 3, 6, 9, 12}, {4, 1, .5, .5, 5, 1, 1, 6, .5, 1, .5, 7}};
    const UpdateStrategy strategies[] = {UpdateStrategy::RightLooking,
        UpdateStrategy::LeftLooking, UpdateStrategy::LeftLooking};
    const SchedulingStrategy schedules[] = {SchedulingStrategy::Level,
        SchedulingStrategy::Level, SchedulingStrategy::SynchronizationFree};
    for (unsigned mode = 0; mode < 3; ++mode) {
        std::ostringstream out, err;
        Symbolic_Matrix cpu(a.n, out, err), gpu(a.n, out, err);
        SymbolicConfig config;
        config.backend = test_backend;
        config.verify = true; config.fallback_to_cpu = false;
        cpu.fill_in(a.rows.data(), a.ptr.data());
        gpu.fill_in(a.rows.data(), a.ptr.data(), config);
        cpu.csr(); gpu.csr();
        cpu.predictLU(a.rows.data(), a.ptr.data(), a.values.data());
        gpu.predictLU(a.rows.data(), a.ptr.data(), a.values.data());
        if (schedules[mode] == SchedulingStrategy::Level) { cpu.leveling(); gpu.leveling(); }
        LUonDevice(cpu, out, err, false, strategies[mode], schedules[mode]);
        LUonDevice(gpu, out, err, false, strategies[mode], schedules[mode]);
        Require(err.str().empty(), err.str());
        for (unsigned p = 0; p < cpu.nnz; ++p)
            Require(std::isfinite(gpu.val[p]) &&
                    std::fabs(cpu.val[p] - gpu.val[p]) <= 1e-12 * (1 + std::fabs(cpu.val[p])),
                    "numeric integration changed the factors");
        unsigned cfgi[2] = {0, 0};
        std::vector<unsigned> permutation(a.n);
        std::vector<int> pivots(a.n);
        for (unsigned i = 0; i < a.n; ++i) permutation[i] = pivots[i] = i;
        SNicsLU identity{};
        identity.cfgi = cfgi;
        identity.row_perm = identity.row_perm_inv = permutation.data();
        identity.col_perm = identity.col_perm_inv = permutation.data();
        identity.pivot = identity.pivot_inv = pivots.data();
        std::vector<REAL> x(a.n), rhs(a.n, 0), transposed_rhs(a.n, 0);
        for (unsigned i = 0; i < a.n; ++i) x[i] = 1. + .25 * i;
        for (unsigned col = 0; col < a.n; ++col)
            for (unsigned p = a.ptr[col]; p < a.ptr[col + 1]; ++p) {
                rhs[a.rows[p]] += a.values[p] * x[col];
                transposed_rhs[col] += a.values[p] * x[a.rows[p]];
            }
        auto solution = gpu.solve(&identity, rhs);
        auto transposed_solution = gpu.solve_CSR(&identity, transposed_rhs);
        for (unsigned i = 0; i < a.n; ++i) {
            Require(std::fabs(solution[i] - x[i]) < 1e-11, "SF symbolic -> solve residual regression");
            Require(std::fabs(transposed_solution[i] - x[i]) < 1e-11, "SF symbolic -> transpose solve regression");
        }
    }
    std::cout << "PASS GPU symbolic integration with RL-level, LL-level and LL-SF\n";
}
#endif

void SFResourceChecks()
{
    Matrix a = RandomMatrix(33, 47, true);
    std::ostringstream out, err;
    Symbolic_Matrix cpu(a.n, out, err), gpu(a.n, out, err);
    cpu.fill_in(a.rows.data(), a.ptr.data()); Prepare(cpu, a);
    SymbolicConfig config;
    config.backend = SymbolicBackend::GPU_SF;
    config.sf_workers = 2; config.pool_entries = 1;
    config.verify = true; config.fallback_to_cpu = false;
    gpu.fill_in(a.rows.data(), a.ptr.data(), config);
    Require(out.str().find("pool retry:") != std::string::npos, "tiny SF arena did not restart");
    Prepare(gpu, a); Compare(cpu, gpu);
    config.max_pool_entries = 1;
    bool failed = false;
    try { gpu.fill_in(a.rows.data(), a.ptr.data(), config); }
    catch (const std::runtime_error &) { failed = true; }
    Require(failed, "strict pool exhaustion silently succeeded");
    Compare(cpu, gpu); // includes old CSR, values and levels: transactional failure
    config.fallback_to_cpu = true;
    gpu.fill_in(a.rows.data(), a.ptr.data(), config);
    Require(out.str().find("GPU symbolic fallback:") != std::string::npos, "pool exhaustion fallback missing");
    Prepare(gpu, a); Compare(cpu, gpu);
    config.pool_entries = config.max_pool_entries = 0;
    Matrix other{a.n, {}, std::vector<unsigned>(size_t(a.n) + 1, 0), {}};
    cpu.fill_in(other.rows.data(), other.ptr.data()); Prepare(cpu, other);
    gpu.fill_in(other.rows.data(), other.ptr.data(), config);
    Require(gpu.val.empty() && gpu.csr_c_idx.empty() && gpu.level_idx.empty(), "different pattern kept caches");
    Prepare(gpu, other); Compare(cpu, gpu);
    std::cout << "PASS SF pool restart, strict failure, fallback and different-pattern reuse\n";
}

void UnavailableChecks()
{
    std::ostringstream out, err;
    unsigned row = 0, ptr[] = {0, 1};
    Symbolic_Matrix matrix(1, out, err);
    SymbolicConfig config;
    config.backend = test_backend;
    matrix.fill_in(&row, ptr, config);
    Require(matrix.symbolic_backend == SymbolicBackend::CPU, "fallback backend identity");
    Require(matrix.sym_r_idx == std::vector<unsigned>({0}) &&
            out.str().find("cpu (fallback)") != std::string::npos, "unavailable GPU fallback");
    config.fallback_to_cpu = false;
    bool failed = false;
    try { matrix.fill_in(&row, ptr, config); }
    catch (const std::runtime_error &) { failed = true; }
    Require(failed, "strict GPU mode silently fell back");
    Require(matrix.sym_r_idx == std::vector<unsigned>({0}) && matrix.nnz == 1,
            "failed GPU analysis damaged the previous pattern");
    Require(matrix.symbolic_backend == SymbolicBackend::CPU, "failed GPU analysis changed backend identity");
    std::cout << "PASS unavailable GPU fallback and strict failure\n";
}

} // namespace

int main(int argc, char **argv)
{
    try {
        bool host_only = false, stress = false;
        for (int i = 1; i < argc; ++i) {
            const std::string arg(argv[i]);
            if (arg == "--host-only") host_only = true;
            else if (arg == "--sf") test_backend = SymbolicBackend::GPU_SF;
            else if (arg == "--stress") stress = true;
            else throw std::invalid_argument("usage: symbolic_gpu_test [--host-only] [--sf] [--stress]");
        }
        GraphSemanticsChecks();
        LevelSemanticsChecks();
        HostChecks();
        // A host-only check must not initialize the GPU runtime: enumeration
        // itself can block when the driver is waiting on an unhealthy card.
        if (host_only) return 0;
        int devices = 0;
        const hcError_t status = hcGetDeviceCount(&devices);
        if (status != hcSuccess || devices == 0) {
            UnavailableChecks();
            std::cout << "SKIP GPU execution: no accessible HTHPCC GPU (exit 77)\n";
            return 77;
        }
        for (unsigned kind = 0; kind < 6; ++kind)
            PatternCase("diagonal-upper-lower-band-block-fanout", StructuredMatrix(65, kind),
                        {64u, 256u}, {1u, 2u, 128u});
        PatternCase("dynamic-predecessors", Matrix{5, {1, 3, 2, 0, 4},
                    {0, 1, 2, 2, 3, 5}, std::vector<double>(5, 1)});
        PatternCase("scalar", Matrix{1, {0}, {0, 1}, {2}});
        PatternCase("empty-columns", Matrix{16, {}, std::vector<unsigned>(17, 0), {}});
        PatternCase("fill-chain", Matrix{4, {3, 2, 1, 0}, {0, 1, 2, 3, 4}, {1, 2, 3, 4}});
        // A path to an illegal output row must still be expanded as an internal
        // node: 3->2->0->4, for which best[0]=3>0 but (4,3) is valid.
        Matrix path{5, {4, 0, 2}, {0, 1, 1, 2, 3, 3}, {1, 2, 3}};
        PatternCase("minimax-path", path);
        // 6->5->0 gives cost 6, then 6->3->2->1->0 improves it to 4.
        // Node 0 must be re-enqueued to discover the legal output row 4.
        PatternCase("epoch-reenqueue", Matrix{8, {7, 4, 0, 1, 2, 0, 5, 3},
                    {0, 2, 3, 4, 5, 5, 6, 8, 8}, std::vector<double>(8, 1)});
        for (unsigned n : {2u, 16u, 32u, 64u, 128u, 257u})
            for (unsigned seed : {13u, 107u, 2026u})
                PatternCase("random-unsorted-missing-diagonal", RandomMatrix(n, seed));
        PatternCase("long-columns", RandomMatrix(257, 1, true));
        // Three scan levels at 64 threads; partial tiles and workspace reuse.
        PatternCase("recursive-scan", Matrix{4097, {}, std::vector<unsigned>(4098, 0), {}},
                    {64u}, {128u});
        if (test_backend == SymbolicBackend::GPU_SF) {
            SFResourceChecks();
            PatternCase("large-n-small-input", Matrix{8193, {}, std::vector<unsigned>(8194, 0), {}},
                        {64u}, {1u, 2u, 1024u});
            if (stress) for (unsigned repeat = 0; repeat < 1000; ++repeat)
                PatternCase("SF-race-stress", RandomMatrix(33, repeat + 71), {64u}, {2u, 128u});
        }
#if defined(GLU_SYMBOLIC_TEST_NUMERIC)
        NumericChecks();
#endif
        std::cout << "PASS CPU/GPU elementwise checks\n";
    } catch (const std::exception &error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
    return 0;
}
