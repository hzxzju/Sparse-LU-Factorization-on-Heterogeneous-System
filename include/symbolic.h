#ifndef _SYMBOLIC_H_
#define _SYMBOLIC_H_
#include <vector>
#include <unordered_set>
#include <iostream>
#include <stdexcept>
#include <string>
#include "type.h"
#include "nicslu.h"

enum class SymbolicBackend { CPU, GPU, GPU_SF };

struct SymbolicConfig
{
    SymbolicBackend backend = SymbolicBackend::CPU;
    unsigned batch_cols = 128;
    unsigned block_size = 256; // 4 native waves of 64 lanes; also supports 1/2 waves.
    bool verify = false;
    bool fallback_to_cpu = true;
    unsigned sf_workers = 0; // 0 = two workers per MP, bounded by memory and n.
    unsigned pool_entries = 0; // Initial capacity of each sparse arena; 0 = heuristic.
    unsigned max_pool_entries = 0; // 0 = memory budget / unsigned index limit.
};

inline SymbolicBackend ParseSymbolicBackend(const std::string &name)
{
    if (name == "cpu") return SymbolicBackend::CPU;
    if (name == "gpu") return SymbolicBackend::GPU;
    if (name == "gpu-sf") return SymbolicBackend::GPU_SF;
    throw std::invalid_argument("symbolic backend must be 'cpu', 'gpu' or 'gpu-sf'");
}

inline const char *SymbolicBackendName(SymbolicBackend backend)
{
    switch (backend) {
    case SymbolicBackend::CPU: return "cpu";
    case SymbolicBackend::GPU: return "gpu";
    case SymbolicBackend::GPU_SF: return "gpu-sf";
    }
    throw std::invalid_argument("Invalid symbolic backend");
}

inline void ValidateSymbolicConfiguration(const SymbolicConfig &config)
{
    if (config.backend != SymbolicBackend::CPU && config.backend != SymbolicBackend::GPU &&
        config.backend != SymbolicBackend::GPU_SF)
        throw std::invalid_argument("Invalid symbolic backend");
    if (config.batch_cols == 0)
        throw std::invalid_argument("symbolic batch_cols must be positive");
    if (config.block_size != 64 && config.block_size != 128 && config.block_size != 256)
        throw std::invalid_argument("symbolic block_size must be 64, 128 or 256");
    if (config.backend == SymbolicBackend::CPU && (config.verify || !config.fallback_to_cpu))
        throw std::invalid_argument("symbolic verification/no-fallback requires a GPU backend");
    if (config.backend != SymbolicBackend::GPU_SF &&
        (config.sf_workers || config.pool_entries || config.max_pool_entries))
        throw std::invalid_argument("symbolic workers/pool options require gpu-sf");
    if (config.max_pool_entries && config.pool_entries > config.max_pool_entries)
        throw std::invalid_argument("symbolic initial pool exceeds maximum pool entries");
}

class Symbolic_Matrix
{
public:
    unsigned int n;
    unsigned int nnz;
    int num_lev;
    // Preferred device after a successful probe. All symbolic arrays remain
    // host-owned, so numeric work can safely switch if this device later fails.
    int gpu_device = -1;
    SymbolicBackend symbolic_backend = SymbolicBackend::CPU; // Last successful fill_in, including fallback.
    std::vector<unsigned> sym_c_ptr;
    std::vector<unsigned> sym_r_idx;
    std::vector<unsigned> csr_r_ptr;
    std::vector<unsigned> csr_c_idx;
    std::vector<unsigned> csr_diag_ptr;
    std::vector<REAL> val;
    std::vector<unsigned> l_col_ptr; // Absolute CSC offsets of structural diagonals.
    std::vector<int> level_idx;
    std::vector<int> level_ptr;

    void predictLU(unsigned*, unsigned*, double*);
    void csr();
    void leveling();
    void fill_in(unsigned *ai, unsigned *ap);
    // GPU graph / synchronization-free adapters: returns the same host CSC arrays as fill_in().
    // Requires symbolic_gpu.cu at link time; the two-argument CPU API is unchanged.
    // Configured backends accept unsorted CSC with no duplicate coordinates.
    // CSR, values and levels are still produced by the existing CPU methods.
    void fill_in(unsigned *ai, unsigned *ap, const SymbolicConfig &config);
    std::vector<REAL> solve(SNicsLU*, const std::vector<REAL> &);
    std::vector<REAL> solve_CSR(SNicsLU*, const std::vector<REAL> &);

#if GLU_DEBUG
    void PrintLevel();
    //ABFT
    std::vector<REAL> CCA;
    void ABFTCalculateCCA();
    void ABFTCheckResult();
#endif

    Symbolic_Matrix(unsigned n, std::ostream &out, std::ostream &err):
        n(n),
        nnz(0),
        num_lev(0),
        m_out(out),
        m_err(err) {};

    Symbolic_Matrix(std::ostream &out, std::ostream &err):
        n(0), nnz(0), num_lev(0), m_out(out), m_err(err) {};

private:
    void reset_derived_structure();
    std::ostream &m_out;
    std::ostream &m_err;
};


#endif
