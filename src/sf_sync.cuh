#ifndef GLU_SF_SYNC_CUH
#define GLU_SF_SYNC_CUH

#include <hc_runtime.h>

// All state lives in device global memory and is naturally unsigned-aligned.
// The production path requires Clang/LLVM lock-free ordered atomics. Do not
// substitute volatile or relaxed legacy atomicAdd for release/acquire.
// HTHPCC lowering and visibility must be checked using tests/sf_sync.cu on
// the target SDK; the supplied runtime guide does not specify this interface.
#if defined(GLU_TEST_CUDA_COMPAT)
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ < 700
#error "The test-only CUDA LL-SF ordered-atomic adapter requires sm_70 or newer"
#endif
#elif !defined(__clang__)
#error "LL-SF requires an HTHPCC Clang compiler with ordered device atomics"
#else
static_assert(__atomic_always_lock_free(sizeof(unsigned), nullptr),
              "LL-SF requires lock-free unsigned device atomics");
#endif

namespace glu_sf {

__device__ inline unsigned fetch_next_column(unsigned *next_col)
{
#if defined(GLU_TEST_CUDA_COMPAT)
    return atomicAdd(next_col, 1u);
#else
    return __atomic_fetch_add(next_col, 1u, __ATOMIC_RELAXED);
#endif
}

__device__ inline unsigned load_done_acquire(unsigned *done)
{
#if defined(GLU_TEST_CUDA_COMPAT)
    unsigned value;
    // PTX device-scope acquire, kept test-only (no CUDA headers in HTHPCC).
    asm volatile("ld.acquire.gpu.global.u32 %0, [%1];"
                 : "=r"(value) : "l"(done) : "memory");
    return value;
#else
    return __atomic_load_n(done, __ATOMIC_ACQUIRE);
#endif
}

__device__ inline void store_done_release(unsigned *done)
{
#if defined(GLU_TEST_CUDA_COMPAT)
    asm volatile("st.release.gpu.global.u32 [%0], %1;"
                 : : "l"(done), "r"(1u) : "memory");
#else
    __atomic_store_n(done, 1u, __ATOMIC_RELEASE);
#endif
}

// Both helpers are collective: every lane in the CTA must participate.
__device__ inline void wait_column_ready(unsigned *done)
{
    if (threadIdx.x == 0) {
        while (load_done_acquire(done) == 0u) {}
    }
    // Propagate the leader's acquire to all lanes before they read the L data.
    __sync_threads();
}

__device__ inline void publish_column_done(unsigned *done)
{
    // Join all lanes' factor writes before the leader publishes the column.
    __sync_threads();
    if (threadIdx.x == 0)
        store_done_release(done);
    // Prevent workspace reuse before the publication step has completed.
    __sync_threads();
}

} // namespace glu_sf
#endif
