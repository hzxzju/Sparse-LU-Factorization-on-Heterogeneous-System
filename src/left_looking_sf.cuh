#ifndef GLU_LEFT_LOOKING_SF_CUH
#define GLU_LEFT_LOOKING_SF_CUH

#include "left_looking.cuh"
#include "sf_sync.cuh"

// Each CTA owns one workspace and processes one column at a time. Natural
// column IDs are issued monotonically, never by blockIdx or by level order.
// All predecessors are < currentCol: the smallest unfinished issued column
// always has completed predecessors. This works even with only one resident
// worker; it does not require all grid blocks to run simultaneously.
__global__ void LL_factorizeSyncFree(
        const unsigned* __restrict__ sym_c_ptr_dev,
        const unsigned* __restrict__ sym_r_idx_dev,
        REAL* __restrict__ val_dev,
        const unsigned* __restrict__ l_col_ptr_dev,
        REAL* __restrict__ tmpMem,
        unsigned* __restrict__ next_col,
        unsigned* __restrict__ done,
        const unsigned n,
        const bool perturb,
        const REAL pert)
{
    __shared__ unsigned currentCol;
    __shared__ REAL pivot;
    REAL *column = tmpMem + static_cast<std::size_t>(blockIdx.x) * n;
    while (true) {
        if (threadIdx.x == 0)
            currentCol = glu_sf::fetch_next_column(next_col);
        __sync_threads();
        if (currentCol >= n)
            return; // uniform across the CTA

        LL_loadColumn(sym_c_ptr_dev, sym_r_idx_dev, val_dev, currentCol, column);
        for (unsigned p = sym_c_ptr_dev[currentCol]; p < l_col_ptr_dev[currentCol]; ++p) {
            const unsigned leftCol = sym_r_idx_dev[p];
            const unsigned leftBegin = l_col_ptr_dev[leftCol] + 1;
            const unsigned leftEnd = sym_c_ptr_dev[leftCol + 1];
            if (leftBegin == leftEnd)
                continue;
            glu_sf::wait_column_ready(done + leftCol);
            LL_updateFromLeft(sym_r_idx_dev, val_dev, leftCol, leftBegin, leftEnd, column);
        }
        LL_finalizeColumn(sym_c_ptr_dev, sym_r_idx_dev, val_dev, currentCol,
                          column, &pivot, perturb, pert);
        glu_sf::publish_column_done(done + currentCol);
    }
}

#endif
