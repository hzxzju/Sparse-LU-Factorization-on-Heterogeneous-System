#ifndef GLU_LEFT_LOOKING_CUH
#define GLU_LEFT_LOOKING_CUH

#include "type.h"
#include <cstddef>
#include <cmath>

// Shared LL arithmetic for level scheduling and synchronization-free workers.
// Every helper below is called collectively by a whole block. Rows in each
// symbolic CSC column must be sorted and unique (CPU fill_in contract).
__device__ inline void LL_loadColumn(
        const unsigned* __restrict__ sym_c_ptr_dev,
        const unsigned* __restrict__ sym_r_idx_dev,
        REAL* __restrict__ val_dev,
        const unsigned currentCol,
        REAL* column)
{
    const unsigned tid = threadIdx.x;
    const unsigned colBegin = sym_c_ptr_dev[currentCol];
    const unsigned colEnd = sym_c_ptr_dev[currentCol + 1];

    // Reload the original target column, including symbolic fill-in zeros.
    // All workspace rows subsequently read belong to this pattern. Reusing a
    // workspace slot therefore needs no clearing after the previous column.
    for (unsigned p = colBegin + tid; p < colEnd; p += blockDim.x)
        column[sym_r_idx_dev[p]] = val_dev[p];
    __sync_threads();
}

__device__ inline void LL_updateFromLeft(
        const unsigned* __restrict__ sym_r_idx_dev,
        REAL* __restrict__ val_dev,
        const unsigned leftCol,
        const unsigned leftBegin,
        const unsigned leftEnd,
        REAL* column)
{
    const REAL upper = column[leftCol];
    for (unsigned q = leftBegin + threadIdx.x; q < leftEnd; q += blockDim.x) {
        const unsigned row = sym_r_idx_dev[q];
        // A single writer per row, in this target column's private workspace.
        column[row] -= val_dev[q] * upper;
    }
    // Complete this contribution before loading the next U(j,k).
    __sync_threads();
}

__device__ inline void LL_finalizeColumn(
        const unsigned* __restrict__ sym_c_ptr_dev,
        const unsigned* __restrict__ sym_r_idx_dev,
        REAL* __restrict__ val_dev,
        const unsigned currentCol,
        REAL* column,
        REAL* shared_pivot,
        const bool perturb,
        const REAL pert)
{
    // Apply the same static diagonal perturbation as the RL kernels, only
    // after the current column has collected every left-column contribution.
    if (threadIdx.x == 0) {
        *shared_pivot = column[currentCol];
        if (perturb && fabs(*shared_pivot) < pert)
            *shared_pivot = pert;
        column[currentCol] = *shared_pivot;
    }
    __sync_threads();

    // Broadcast the finalized pivot explicitly instead of allowing individual
    // lanes to reuse a previously loaded workspace value during normalization.
    for (unsigned p = sym_c_ptr_dev[currentCol] + threadIdx.x;
         p < sym_c_ptr_dev[currentCol + 1]; p += blockDim.x) {
        const unsigned row = sym_r_idx_dev[p];
        val_dev[p] = row > currentCol ? column[row] / *shared_pivot : column[row];
    }
}

// One block owns one target column. Earlier levels finalized its L inputs.
__global__ void LL_factorizeColumns(
        const unsigned* __restrict__ sym_c_ptr_dev,
        const unsigned* __restrict__ sym_r_idx_dev,
        REAL* __restrict__ val_dev,
        const unsigned* __restrict__ l_col_ptr_dev,
        const int* __restrict__ level_idx_dev,
        REAL* __restrict__ tmpMem,
        const unsigned n,
        const unsigned levelHead,
        const unsigned inLevPos,
        const bool perturb,
        const REAL pert)
{
    const unsigned currentCol = level_idx_dev[levelHead + inLevPos + blockIdx.x];
    REAL *column = tmpMem + static_cast<std::size_t>(blockIdx.x) * n;
    __shared__ REAL pivot;
    LL_loadColumn(sym_c_ptr_dev, sym_r_idx_dev, val_dev, currentCol, column);
    for (unsigned p = sym_c_ptr_dev[currentCol]; p < l_col_ptr_dev[currentCol]; ++p) {
        const unsigned leftCol = sym_r_idx_dev[p];
        const unsigned leftBegin = l_col_ptr_dev[leftCol] + 1;
        const unsigned leftEnd = sym_c_ptr_dev[leftCol + 1];
        if (leftBegin == leftEnd)
            continue;
        LL_updateFromLeft(sym_r_idx_dev, val_dev, leftCol, leftBegin, leftEnd, column);
    }
    LL_finalizeColumn(sym_c_ptr_dev, sym_r_idx_dev, val_dev, currentCol,
                      column, &pivot, perturb, pert);
}

#endif
