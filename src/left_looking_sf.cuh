#ifndef GLU_LEFT_LOOKING_SF_CUH
#define GLU_LEFT_LOOKING_SF_CUH

#include "left_looking.cuh"
#include "sf_sync.cuh"

namespace glu_sf {

// Keep native wave boundaries: 24 active search lanes in each 64-lane wave.
// The host still launches the original (at most 256-thread) LL block.
constexpr unsigned ll_wave_size = 64;
constexpr unsigned ll_search_lanes = 24;
constexpr unsigned ll_search_stride = 64;
constexpr unsigned ll_max_block_threads = 256;

} // namespace glu_sf

// Collective over the CTA. For one target row, each wave owns one predecessor
// at a time, then advances by the number of waves in the block. Its 24 lanes
// binary-search disjoint, sorted stripes of that predecessor's L column.
// Each lane covers multiple stripe offsets so a stride of 64 still covers
// every entry when only 24 lanes participate.
__device__ inline void LL_updateElementFromLeftWaves(
        const unsigned* __restrict__ sym_c_ptr_dev,
        const unsigned* __restrict__ sym_r_idx_dev,
        const REAL* __restrict__ val_dev,
        const unsigned* __restrict__ l_col_ptr_dev,
        unsigned* __restrict__ done,
        const unsigned predecessorBegin,
        const unsigned predecessorEnd,
        const unsigned row,
        REAL* column,
        REAL* partials)
{
    const unsigned wave = threadIdx.x / glu_sf::ll_wave_size;
    const unsigned lane = threadIdx.x % glu_sf::ll_wave_size;
    const unsigned waves = blockDim.x / glu_sf::ll_wave_size;
    REAL sum = 0;
    if (lane < glu_sf::ll_search_lanes) {
        for (unsigned p = predecessorBegin + wave; p < predecessorEnd; p += waves) {
            const unsigned leftCol = sym_r_idx_dev[p];
            // Upper entries must consume only already-computed U(k,j).
            if (leftCol >= row)
                break;
            const unsigned begin = l_col_ptr_dev[leftCol] + 1;
            const unsigned size = sym_c_ptr_dev[leftCol + 1] - begin;
            for (unsigned stripe = lane;
                 stripe < glu_sf::ll_search_stride && stripe < size;
                 stripe += glu_sf::ll_search_lanes) {
                const unsigned count = 1 + (size - 1 - stripe) / glu_sf::ll_search_stride;
                unsigned lo = 0, hi = count;
                while (lo < hi) {
                    const unsigned mid = lo + (hi - lo) / 2;
                    const unsigned q = begin + stripe + mid * glu_sf::ll_search_stride;
                    if (sym_r_idx_dev[q] < row)
                        lo = mid + 1;
                    else
                        hi = mid;
                }
                if (lo < count) {
                    const unsigned q = begin + stripe + lo * glu_sf::ll_search_stride;
                    if (sym_r_idx_dev[q] == row) {
                        // The immutable structure can be searched before readiness.
                        // Only the matching lane reads values, after its own acquire;
                        // no CTA barrier is allowed inside this wave-local loop.
                        while (glu_sf::load_done_acquire(done + leftCol) == 0u) {}
                        sum += val_dev[q] * column[leftCol];
                    }
                }
            }
        }
        partials[threadIdx.x] = sum;
    }
    __sync_threads();
    if (threadIdx.x == 0) {
        // A fixed reduction order avoids racing updates to the same workspace
        // element. Inactive lanes do not supply partials.
        REAL total = 0;
        for (unsigned w = 0; w < waves; ++w)
            for (unsigned l = 0; l < glu_sf::ll_search_lanes; ++l)
                total += partials[w * glu_sf::ll_wave_size + l];
        column[row] -= total;
    }
    // Publish U(row,j) locally before any later row consumes it.
    __sync_threads();
}

// Each CTA owns one workspace and processes one column at a time. Natural
// column IDs are issued monotonically, never by blockIdx or by level order.
// All predecessors are < currentCol: the smallest unfinished issued column
// always has completed predecessors. This works even with only one resident
// worker; it does not require all grid blocks to run simultaneously.
template <bool WaveUpdate>
__device__ inline void LL_runSyncFreeWorker(
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
        const unsigned workspace_id,
        unsigned* currentCol,
        REAL* pivot)
{
    __shared__ REAL partials[glu_sf::ll_max_block_threads];
    REAL *column = tmpMem + static_cast<std::size_t>(workspace_id) * n;
    while (true) {
        if (threadIdx.x == 0)
            *currentCol = glu_sf::fetch_next_column(next_col);
        __sync_threads();
        if (*currentCol >= n)
            return; // uniform across the CTA

        LL_loadColumn(sym_c_ptr_dev, sym_r_idx_dev, val_dev, *currentCol, column);
        const unsigned colBegin = sym_c_ptr_dev[*currentCol];
        const unsigned predecessorEnd = l_col_ptr_dev[*currentCol];
        if (WaveUpdate) {
            // Sorted row order preserves dependencies inside U(:,j).
            for (unsigned p = colBegin; p < sym_c_ptr_dev[*currentCol + 1]; ++p) {
                LL_updateElementFromLeftWaves(sym_c_ptr_dev, sym_r_idx_dev, val_dev,
                                             l_col_ptr_dev, done, colBegin,
                                             predecessorEnd, sym_r_idx_dev[p],
                                             column, partials);
            }
        } else {
            // Original LL-SF: the entire block applies one predecessor at a
            // time. Complete its contribution before reading the next U(k,j).
            for (unsigned p = colBegin; p < predecessorEnd; ++p) {
                const unsigned leftCol = sym_r_idx_dev[p];
                const unsigned begin = l_col_ptr_dev[leftCol] + 1;
                const unsigned end = sym_c_ptr_dev[leftCol + 1];
                if (begin == end) continue;
                glu_sf::wait_column_ready(done + leftCol);
                LL_updateFromLeft(sym_r_idx_dev, val_dev, leftCol, begin, end, column);
            }
        }
        LL_finalizeColumn(sym_c_ptr_dev, sym_r_idx_dev, val_dev, *currentCol,
                          column, pivot, perturb, pert);
        glu_sf::publish_column_done(done + *currentCol);
    }
}

template <bool WaveUpdate = false>
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
    LL_runSyncFreeWorker<WaveUpdate>(sym_c_ptr_dev, sym_r_idx_dev, val_dev, l_col_ptr_dev,
                         tmpMem, next_col, done, n, perturb, pert,
                         blockIdx.x, &currentCol, &pivot);
}

#endif
