#ifndef GLU_SF_SYNC_CUH
#define GLU_SF_SYNC_CUH

#include <hc_runtime.h>

// All state lives in device global memory and is naturally unsigned-aligned.
// Use the SDK's ordered device-scope atomics, as its native atomicAdd does.
// Generic __atomic_* emits system-scope IR on this SDK. Do not substitute
// volatile or relaxed legacy atomicAdd for release/acquire publication.
// Validate device progress and visibility using tests/sf_sync.cu on the GPU.
#if defined(GLU_TEST_CUDA_COMPAT)
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ < 700
#error "The test-only CUDA LL-SF ordered-atomic adapter requires sm_70 or newer"
#endif
#elif !defined(__clang__)
#error "LL-SF requires an HTHPCC Clang compiler with ordered device atomics"
#elif !__has_builtin(__hpcc_atomic_fetch_add) || \
      !__has_builtin(__hpcc_atomic_load) || !__has_builtin(__hpcc_atomic_store)
#error "LL-SF requires HTHPCC ordered device-scope atomic builtins"
#endif

namespace glu_sf {

__device__ inline unsigned fetch_next_column(unsigned *next_col)
{
#if defined(GLU_TEST_CUDA_COMPAT)
    return atomicAdd(next_col, 1u);
#else
    return __hpcc_atomic_fetch_add(next_col, 1u, __ATOMIC_RELAXED, memory_scope_device);
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
    return __hpcc_atomic_load(done, __ATOMIC_ACQUIRE, memory_scope_device);
#endif
}

__device__ inline void store_done_release(unsigned *done)
{
#if defined(GLU_TEST_CUDA_COMPAT)
    asm volatile("st.release.gpu.global.u32 [%0], %1;"
                 : : "l"(done), "r"(1u) : "memory");
#else
    __hpcc_atomic_store(done, 1u, __ATOMIC_RELEASE, memory_scope_device);
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
