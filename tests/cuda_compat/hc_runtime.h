#ifndef GLU_TEST_CUDA_COMPAT_H
#define GLU_TEST_CUDA_COMPAT_H
#define GLU_TEST_CUDA_COMPAT 1

// TEST ONLY: run the unchanged numeric.cu through NVIDIA CUDA when HTHPCC is
// unavailable. This validates arithmetic and block synchronization, not HTHPCC
// compiler/runtime compatibility. Never add this directory to production builds.
#include <cuda_runtime.h>
using hcError_t = cudaError_t;
using hcEvent_t = cudaEvent_t;
using hcStream_t = cudaStream_t;
struct hcDeviceProp_t : cudaDeviceProp { int waveSize; };
inline hcError_t hcGetDeviceProperties(hcDeviceProp_t *props, int device)
{
    const cudaError_t status = cudaGetDeviceProperties(props, device);
    // GLU partitions updates into groups of 64 lanes. Explicit block barriers
    // make its kernels safe on CUDA too; this is not the hardware warp width.
    props->waveSize = 64;
    return status;
}
#define __sync_threads __syncthreads
#define hcSuccess cudaSuccess
#define hcGetDeviceCount cudaGetDeviceCount
#define hcSetDevice cudaSetDevice
#define hcEventCreate cudaEventCreate
#define hcEventRecord cudaEventRecord
#define hcEventSynchronize cudaEventSynchronize
#define hcEventElapsedTime cudaEventElapsedTime
#define hcEventDestroy cudaEventDestroy
#define hcMalloc cudaMalloc
#define hcFree cudaFree
#define hcMemcpy cudaMemcpy
#define hcMemcpyHostToDevice cudaMemcpyHostToDevice
#define hcMemcpyDeviceToHost cudaMemcpyDeviceToHost
#define hcMemset cudaMemset
#define hcStreamCreate cudaStreamCreate
#define hcStreamDestroy cudaStreamDestroy
#define hcDeviceSynchronize cudaDeviceSynchronize
#define hcGetLastError cudaGetLastError

#endif
