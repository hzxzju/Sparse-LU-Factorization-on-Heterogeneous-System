#ifndef GLU_NUMERIC_SM_PADDING_CUH
#define GLU_NUMERIC_SM_PADDING_CUH

#include "left_looking_sf.cuh"

namespace glu_numeric_padding {

__device__ inline void complete_worker(unsigned *completed)
{
#if defined(GLU_TEST_CUDA_COMPAT)
    unsigned previous;
    asm volatile("atom.acq_rel.gpu.global.add.u32 %0, [%1], 1;"
                 : "=r"(previous) : "l"(completed) : "memory");
#else
    __hpcc_atomic_fetch_add(completed, 1u, __ATOMIC_ACQ_REL, memory_scope_device);
#endif
}

} // namespace glu_numeric_padding

// Cooperative grid: exactly one resident block per SM, verified by the host.
// Only compute_workers blocks own a dense column workspace. All blocks remain
// resident until every compute worker has exhausted its original task loop.
__global__ void LL_factorizeSyncFreeSmPadding(
        const unsigned* __restrict__ sym_c_ptr_dev,
        const unsigned* __restrict__ sym_r_idx_dev,
        REAL* __restrict__ val_dev,
        const unsigned* __restrict__ l_col_ptr_dev,
        REAL* __restrict__ tmpMem,
        unsigned* __restrict__ next_col,
        unsigned* __restrict__ done,
        const unsigned n,
        const bool perturb,
        const REAL pert,
        const unsigned compute_workers,
        unsigned* __restrict__ completed_workers)
{
    extern __shared__ volatile unsigned char sm_reservation[];
    if (threadIdx.x == 0)
        sm_reservation[0] = 0; // Keep the dynamic shared-memory reservation explicit.
    __shared__ unsigned currentCol;
    __shared__ REAL pivot;
    if (blockIdx.x < compute_workers) {
        // Compact workspace IDs are [0, compute_workers); padding blocks must
        // never access tmpMem or claim a column from next_col.
        LL_runSyncFreeWorker(sym_c_ptr_dev, sym_r_idx_dev, val_dev, l_col_ptr_dev,
                             tmpMem, next_col, done, n, perturb, pert,
                             blockIdx.x, &currentCol, &pivot);
        __sync_threads();
        if (threadIdx.x == 0)
            glu_numeric_padding::complete_worker(completed_workers);
    }
    if (threadIdx.x == 0) {
        while (glu_sf::load_done_acquire(completed_workers) != compute_workers) {}
    }
    __sync_threads();
}

#endif
