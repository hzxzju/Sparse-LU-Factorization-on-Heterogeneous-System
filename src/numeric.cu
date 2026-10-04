#include <hc_runtime.h>
#include <iostream>
#include "symbolic.h"
#include <cmath>
#include <limits>

using namespace std;


__global__ void RL(
        const unsigned* __restrict__ sym_c_ptr_dev,
        const unsigned* __restrict__ sym_r_idx_dev,
        REAL* __restrict__ val_dev,
        const unsigned* __restrict__ l_col_ptr_dev,
        const unsigned* __restrict__ csr_r_ptr_dev,
        const unsigned* __restrict__ csr_c_idx_dev,
        const unsigned* __restrict__ csr_diag_ptr_dev,
        const int* __restrict__ level_idx_dev,
        REAL* __restrict__ tmpMem,
        const unsigned n,
        const int levelHead,
        const int inLevPos)
{
    const int tid = threadIdx.x;
    const int bid = blockIdx.x;
    // HTHPCC wave width is 64 (CUDA warp width is 32).
    const int wid = threadIdx.x / 64;

    const unsigned currentCol = level_idx_dev[levelHead+inLevPos+bid];
    const unsigned currentLColSize = sym_c_ptr_dev[currentCol + 1] - l_col_ptr_dev[currentCol] - 1;
    const unsigned currentLPos = l_col_ptr_dev[currentCol] + tid + 1;

    //update current col

    int offset = 0;
    while (currentLColSize > offset)
    {
        if (tid + offset < currentLColSize)
        {
            unsigned ridx = sym_r_idx_dev[currentLPos + offset];

            val_dev[currentLPos + offset] /= val_dev[l_col_ptr_dev[currentCol]];
            tmpMem[bid*n + ridx]= val_dev[currentLPos + offset];
        }
        offset += blockDim.x;
    }
    __sync_threads();

    //broadcast to submatrix
    const unsigned subColPos = csr_diag_ptr_dev[currentCol] + wid + 1;
    const unsigned subMatSize = csr_r_ptr_dev[currentCol + 1] - csr_diag_ptr_dev[currentCol] - 1;
    unsigned subCol;
    const int tidInWave = threadIdx.x % 64;
    unsigned subColElem = 0;

    int woffset = 0;
    while (subMatSize > woffset)
    {
        if (wid + woffset < subMatSize)
        {
            offset = 0;
            subCol = csr_c_idx_dev[subColPos + woffset];
            // CUDA relied on implicit warp lockstep to broadcast s[wid]. HTHPCC
            // does not promise CUDA's 32-lane warp behavior. Each lane locates
            // the pivot independently, avoiding cross-lane shared-memory reads.
            REAL pivot = 0;
            for (unsigned p = sym_c_ptr_dev[subCol]; p < sym_c_ptr_dev[subCol + 1]; ++p)
            {
                if (sym_r_idx_dev[p] == currentCol)
                {
                    pivot = val_dev[p];
                    break;
                }
            }
            while(offset < sym_c_ptr_dev[subCol + 1] - sym_c_ptr_dev[subCol])
            {
                if (tidInWave + offset < sym_c_ptr_dev[subCol + 1] - sym_c_ptr_dev[subCol])
                {

                    subColElem = sym_c_ptr_dev[subCol] + tidInWave + offset;
                    unsigned ridx = sym_r_idx_dev[subColElem];

                    if (ridx > currentCol)
                    {
                        //elem in currentCol same row with subColElem might be 0, so
                        //clearing tmpMem is necessary
                        atomicAdd(&val_dev[subColElem], -tmpMem[ridx+n*bid]*pivot);
                    }
                }
                offset += 64;
            }
        }
        woffset += blockDim.x/64;
    }

    __sync_threads();
    //Clear tmpMem
    offset = 0;
    while (currentLColSize > offset)
    {
        if (tid + offset < currentLColSize)
        {
            unsigned ridx = sym_r_idx_dev[currentLPos + offset];
            tmpMem[bid*n + ridx]= 0;
        }
        offset += blockDim.x;
    }
}

__global__ void RL_perturb(
        const unsigned* __restrict__ sym_c_ptr_dev,
        const unsigned* __restrict__ sym_r_idx_dev,
        REAL* __restrict__ val_dev,
        const unsigned* __restrict__ l_col_ptr_dev,
        const unsigned* __restrict__ csr_r_ptr_dev,
        const unsigned* __restrict__ csr_c_idx_dev,
        const unsigned* __restrict__ csr_diag_ptr_dev,
        const int* __restrict__ level_idx_dev,
        REAL* __restrict__ tmpMem,
        const unsigned n,
        const int levelHead,
        const int inLevPos,
        const REAL pert)
{
    const int tid = threadIdx.x;
    const int bid = blockIdx.x;
    // HTHPCC wave width is 64 (CUDA warp width is 32).
    const int wid = threadIdx.x / 64;

    const unsigned currentCol = level_idx_dev[levelHead+inLevPos+bid];
    const unsigned currentLColSize = sym_c_ptr_dev[currentCol + 1] - l_col_ptr_dev[currentCol] - 1;
    const unsigned currentLPos = l_col_ptr_dev[currentCol] + tid + 1;

    //update current col

    // One lane updates the diagonal pivot; the block barrier makes the value
    // visible before any lane uses it, including when this column has no L entries.
    if (tid == 0 && fabs(val_dev[l_col_ptr_dev[currentCol]]) < pert)
        val_dev[l_col_ptr_dev[currentCol]] = pert;
    __sync_threads();

    int offset = 0;
    while (currentLColSize > offset)
    {
        if (tid + offset < currentLColSize)
        {
            unsigned ridx = sym_r_idx_dev[currentLPos + offset];
            val_dev[currentLPos + offset] /= val_dev[l_col_ptr_dev[currentCol]];
            tmpMem[bid*n + ridx]= val_dev[currentLPos + offset];
        }
        offset += blockDim.x;
    }
    __sync_threads();

    //broadcast to submatrix
    const unsigned subColPos = csr_diag_ptr_dev[currentCol] + wid + 1;
    const unsigned subMatSize = csr_r_ptr_dev[currentCol + 1] - csr_diag_ptr_dev[currentCol] - 1;
    unsigned subCol;
    const int tidInWave = threadIdx.x % 64;
    unsigned subColElem = 0;

    int woffset = 0;
    while (subMatSize > woffset)
    {
        if (wid + woffset < subMatSize)
        {
            offset = 0;
            subCol = csr_c_idx_dev[subColPos + woffset];
            // See RL: independent pivot search removes CUDA warp-lockstep semantics.
            REAL pivot = 0;
            for (unsigned p = sym_c_ptr_dev[subCol]; p < sym_c_ptr_dev[subCol + 1]; ++p)
            {
                if (sym_r_idx_dev[p] == currentCol)
                {
                    pivot = val_dev[p];
                    break;
                }
            }
            while(offset < sym_c_ptr_dev[subCol + 1] - sym_c_ptr_dev[subCol])
            {
                if (tidInWave + offset < sym_c_ptr_dev[subCol + 1] - sym_c_ptr_dev[subCol])
                {

                    subColElem = sym_c_ptr_dev[subCol] + tidInWave + offset;
                    unsigned ridx = sym_r_idx_dev[subColElem];

                    if (ridx > currentCol)
                    {
                        //elem in currentCol same row with subColElem might be 0, so
                        //clearing tmpMem is necessary
                        atomicAdd(&val_dev[subColElem], -tmpMem[ridx+n*bid]*pivot);
                    }
                }
                offset += 64;
            }
        }
        woffset += blockDim.x/64;
    }

    __sync_threads();
    //Clear tmpMem
    offset = 0;
    while (currentLColSize > offset)
    {
        if (tid + offset < currentLColSize)
        {
            unsigned ridx = sym_r_idx_dev[currentLPos + offset];
            tmpMem[bid*n + ridx]= 0;
        }
        offset += blockDim.x;
    }
}

__global__ void RL_onecol_factorizeCurrentCol(
        const unsigned* __restrict__ sym_c_ptr_dev,
        const unsigned* __restrict__ sym_r_idx_dev,
        REAL* __restrict__ val_dev,
        const unsigned* __restrict__ l_col_ptr_dev,
        const unsigned currentCol,
        REAL* __restrict__ tmpMem,
        const int stream,
        const unsigned n)
{
    const int tid = threadIdx.x;

    const unsigned currentLColSize = sym_c_ptr_dev[currentCol + 1] - l_col_ptr_dev[currentCol] - 1;
    const unsigned currentLPos = l_col_ptr_dev[currentCol] + tid + 1;

    //update current col

    int offset = 0;
    while (currentLColSize > offset)
    {
        if (tid + offset < currentLColSize)
        {
            unsigned ridx = sym_r_idx_dev[currentLPos + offset];

            val_dev[currentLPos + offset] /= val_dev[l_col_ptr_dev[currentCol]];
            tmpMem[stream * n + ridx]= val_dev[currentLPos + offset];
        }
        offset += blockDim.x;
    }
}

__global__ void RL_onecol_factorizeCurrentCol_perturb(
        const unsigned* __restrict__ sym_c_ptr_dev,
        const unsigned* __restrict__ sym_r_idx_dev,
        REAL* __restrict__ val_dev,
        const unsigned* __restrict__ l_col_ptr_dev,
        const unsigned currentCol,
        REAL* __restrict__ tmpMem,
        const int stream,
        const unsigned n,
        const REAL pert)
{
    const int tid = threadIdx.x;

    const unsigned currentLColSize = sym_c_ptr_dev[currentCol + 1] - l_col_ptr_dev[currentCol] - 1;
    const unsigned currentLPos = l_col_ptr_dev[currentCol] + tid + 1;

    //update current col

    // One lane updates the diagonal pivot; synchronize before peer reads.
    if (tid == 0 && fabs(val_dev[l_col_ptr_dev[currentCol]]) < pert)
        val_dev[l_col_ptr_dev[currentCol]] = pert;
    __sync_threads();

    int offset = 0;
    while (currentLColSize > offset)
    {
        if (tid + offset < currentLColSize)
        {
            unsigned ridx = sym_r_idx_dev[currentLPos + offset];

            val_dev[currentLPos + offset] /= val_dev[l_col_ptr_dev[currentCol]];
            tmpMem[stream * n + ridx]= val_dev[currentLPos + offset];
        }
        offset += blockDim.x;
    }
}

__global__ void RL_onecol_updateSubmat(
        const unsigned* __restrict__ sym_c_ptr_dev,
        const unsigned* __restrict__ sym_r_idx_dev,
        REAL* __restrict__ val_dev,
        const unsigned* __restrict__ csr_c_idx_dev,
        const unsigned* __restrict__ csr_diag_ptr_dev,
        const unsigned currentCol,
        REAL* __restrict__ tmpMem,
        const int stream,
        const unsigned n)
{
    const int tid = threadIdx.x;
    const int bid = blockIdx.x;
    __shared__ REAL s;

    //broadcast to submatrix
    const unsigned subColPos = csr_diag_ptr_dev[currentCol] + bid + 1;
    const unsigned subCol = csr_c_idx_dev[subColPos];
    unsigned subColElem = 0;

    // Find the pivot once, then publish it to every thread in this block.
    // A block barrier avoids depending on CUDA warp lockstep or on the pivot
    // being encountered in the same chunk as the rows updated below.
    if (tid == 0) {
        s = 0;
        for (unsigned p = sym_c_ptr_dev[subCol]; p < sym_c_ptr_dev[subCol + 1]; ++p) {
            if (sym_r_idx_dev[p] == currentCol) {
                s = val_dev[p];
                break;
            }
        }
    }
    __sync_threads();
    int offset = 0;
    while(offset < sym_c_ptr_dev[subCol + 1] - sym_c_ptr_dev[subCol])
    {
        bool active = (tid + offset < sym_c_ptr_dev[subCol + 1] - sym_c_ptr_dev[subCol]);
        unsigned ridx = 0;
        if (active)
        {
            subColElem = sym_c_ptr_dev[subCol] + tid + offset;
            ridx = sym_r_idx_dev[subColElem];
        }
        if (active && ridx > currentCol)
        {
            atomicAdd(&val_dev[subColElem], -tmpMem[stream * n + ridx] * s);
        }
        offset += blockDim.x;
    }
}

__global__ void RL_onecol_cleartmpMem(
        const unsigned* __restrict__ sym_c_ptr_dev,
        const unsigned* __restrict__ sym_r_idx_dev,
        const unsigned* __restrict__ l_col_ptr_dev,
        const unsigned currentCol,
        REAL* __restrict__ tmpMem,
        const int stream,
        const unsigned n)
{
    const int tid = threadIdx.x;

    const unsigned currentLColSize = sym_c_ptr_dev[currentCol + 1] - l_col_ptr_dev[currentCol] - 1;
    const unsigned currentLPos = l_col_ptr_dev[currentCol] + tid + 1;

    unsigned offset = 0;
    while (currentLColSize > offset)
    {
        if (tid + offset < currentLColSize)
        {
            unsigned ridx = sym_r_idx_dev[currentLPos + offset];
            tmpMem[stream * n + ridx]= 0;
        }
        offset += blockDim.x;
    }
}

void LUonDevice(Symbolic_Matrix &A_sym, ostream &out, ostream &err, bool PERTURB)
{
    if (A_sym.n == 0 || A_sym.nnz == 0) {
        err << "Matrix is empty; skipping GPU factorization." << endl;
        return;
    }

    auto hcCheck = [&](hcError_t status, const char *op) -> bool {
        if (status != hcSuccess) {
            // The guide documents hcError_t/hcSuccess, but not an error-string
            // formatter; print the numeric runtime status without guessing one.
            err << op << " failed with HTHPCC status " << static_cast<int>(status) << endl;
            return false;
        }
        return true;
    };

    unsigned n = A_sym.n;
    unsigned nnz = A_sym.nnz;
    unsigned num_lev = A_sym.num_lev;

    int deviceCount = 0;
    if (!hcCheck(hcGetDeviceCount(&deviceCount), "hcGetDeviceCount"))
        return;
    if (deviceCount <= 0) {
        err << "No HTHPCC GPU detected." << endl;
        return;
    }

    int dev = 0;
    hcDeviceProp_t deviceProp;
    if (!hcCheck(hcGetDeviceProperties(&deviceProp, dev), "hcGetDeviceProperties"))
        return;
    if (deviceProp.waveSize != 64) {
        err << "Expected HTHPCC waveSize 64, got " << deviceProp.waveSize << endl;
        return;
    }
    // Keep all blocks wave-aligned and within the selected device's limit.
    const int maxWaveAlignedThreads = (deviceProp.maxThreadsPerBlock / 64) * 64;
    if (maxWaveAlignedThreads < 64) {
        err << "Device maxThreadsPerBlock is smaller than one 64-thread wave." << endl;
        return;
    }
    const int onecolBlockThreads = (maxWaveAlignedThreads < 1024)
        ? maxWaveAlignedThreads : 1024;
    if (!hcCheck(hcSetDevice(dev), "hcSetDevice"))
        return;
    out << "Device " << dev << ": " << deviceProp.name << " has been selected." << endl;

    hcEvent_t start = nullptr, stop = nullptr;
    unsigned *sym_c_ptr_dev = nullptr, *sym_r_idx_dev = nullptr, *l_col_ptr_dev = nullptr;
    REAL *val_dev = nullptr;
    unsigned *csr_r_ptr_dev = nullptr, *csr_c_idx_dev = nullptr, *csr_diag_ptr_dev = nullptr;
    int *level_idx_dev = nullptr;
    REAL *tmpMem = nullptr;
    float time = 0.0f;

    constexpr int Nstreams = 16;
    hcStream_t streams[Nstreams];
    bool stream_created[Nstreams] = {false};

    auto cleanup = [&]() {
        for (int j = 0; j < Nstreams; ++j) {
            if (stream_created[j])
                hcStreamDestroy(streams[j]);
        }
        if (tmpMem != nullptr)
            hcFree(tmpMem);
        if (sym_c_ptr_dev != nullptr)
            hcFree(sym_c_ptr_dev);
        if (sym_r_idx_dev != nullptr)
            hcFree(sym_r_idx_dev);
        if (val_dev != nullptr)
            hcFree(val_dev);
        if (l_col_ptr_dev != nullptr)
            hcFree(l_col_ptr_dev);
        if (csr_c_idx_dev != nullptr)
            hcFree(csr_c_idx_dev);
        if (csr_r_ptr_dev != nullptr)
            hcFree(csr_r_ptr_dev);
        if (csr_diag_ptr_dev != nullptr)
            hcFree(csr_diag_ptr_dev);
        if (level_idx_dev != nullptr)
            hcFree(level_idx_dev);
        if (start != nullptr)
            hcEventDestroy(start);
        if (stop != nullptr)
            hcEventDestroy(stop);
    };

#define HC_RETURN_ON_ERR(call, op_name) \
    do { \
        if (!hcCheck((call), (op_name))) { \
            cleanup(); \
            return; \
        } \
    } while (0)

    HC_RETURN_ON_ERR(hcEventCreate(&start), "hcEventCreate(start)");
    HC_RETURN_ON_ERR(hcEventCreate(&stop), "hcEventCreate(stop)");
    HC_RETURN_ON_ERR(hcEventRecord(start), "hcEventRecord(start)");

    // hcMalloc allocations are SVM-accessible from host and device according
    // to the guide; data is still copied explicitly to preserve ownership flow.
    HC_RETURN_ON_ERR(hcMalloc((void**)&sym_c_ptr_dev, (n + 1) * sizeof(unsigned)), "hcMalloc(sym_c_ptr_dev)");
    HC_RETURN_ON_ERR(hcMalloc((void**)&sym_r_idx_dev, nnz * sizeof(unsigned)), "hcMalloc(sym_r_idx_dev)");
    HC_RETURN_ON_ERR(hcMalloc((void**)&val_dev, nnz * sizeof(REAL)), "hcMalloc(val_dev)");
    HC_RETURN_ON_ERR(hcMalloc((void**)&l_col_ptr_dev, n * sizeof(unsigned)), "hcMalloc(l_col_ptr_dev)");
    HC_RETURN_ON_ERR(hcMalloc((void**)&csr_r_ptr_dev, (n + 1) * sizeof(unsigned)), "hcMalloc(csr_r_ptr_dev)");
    HC_RETURN_ON_ERR(hcMalloc((void**)&csr_c_idx_dev, nnz * sizeof(unsigned)), "hcMalloc(csr_c_idx_dev)");
    HC_RETURN_ON_ERR(hcMalloc((void**)&csr_diag_ptr_dev, n * sizeof(unsigned)), "hcMalloc(csr_diag_ptr_dev)");
    HC_RETURN_ON_ERR(hcMalloc((void**)&level_idx_dev, n * sizeof(int)), "hcMalloc(level_idx_dev)");

    // hcMemcpy is blocking in HTHPCC; each upload completes before kernels use it.
    HC_RETURN_ON_ERR(hcMemcpy(sym_c_ptr_dev, &(A_sym.sym_c_ptr[0]), (n + 1) * sizeof(unsigned), hcMemcpyHostToDevice),
        "hcMemcpy(sym_c_ptr_dev)");
    HC_RETURN_ON_ERR(hcMemcpy(sym_r_idx_dev, &(A_sym.sym_r_idx[0]), nnz * sizeof(unsigned), hcMemcpyHostToDevice),
        "hcMemcpy(sym_r_idx_dev)");
    HC_RETURN_ON_ERR(hcMemcpy(val_dev, &(A_sym.val[0]), nnz * sizeof(REAL), hcMemcpyHostToDevice), "hcMemcpy(val_dev)");
    HC_RETURN_ON_ERR(hcMemcpy(l_col_ptr_dev, &(A_sym.l_col_ptr[0]), n * sizeof(unsigned), hcMemcpyHostToDevice),
        "hcMemcpy(l_col_ptr_dev)");
    HC_RETURN_ON_ERR(hcMemcpy(csr_r_ptr_dev, &(A_sym.csr_r_ptr[0]), (n + 1) * sizeof(unsigned), hcMemcpyHostToDevice),
        "hcMemcpy(csr_r_ptr_dev)");
    HC_RETURN_ON_ERR(hcMemcpy(csr_c_idx_dev, &(A_sym.csr_c_idx[0]), nnz * sizeof(unsigned), hcMemcpyHostToDevice),
        "hcMemcpy(csr_c_idx_dev)");
    HC_RETURN_ON_ERR(hcMemcpy(csr_diag_ptr_dev, &(A_sym.csr_diag_ptr[0]), n * sizeof(unsigned), hcMemcpyHostToDevice),
        "hcMemcpy(csr_diag_ptr_dev)");
    HC_RETURN_ON_ERR(hcMemcpy(level_idx_dev, &(A_sym.level_idx[0]), n * sizeof(int), hcMemcpyHostToDevice),
        "hcMemcpy(level_idx_dev)");

    for (int j = 0; j < Nstreams; ++j) {
        HC_RETURN_ON_ERR(hcStreamCreate(&streams[j]), "hcStreamCreate");
        stream_created[j] = true;
    }

    const unsigned first_level_cols = (A_sym.level_ptr.size() > 1) ?
        static_cast<unsigned>(A_sym.level_ptr[1] - A_sym.level_ptr[0]) : 1u;
    // hcMemGetInfo is not documented in the supplied runtime guide. Bound the
    // temporary matrix to one 16-stream batch; hcMalloc reports allocation failure.
    unsigned TMPMEMNUM = (first_level_cols < Nstreams) ? first_level_cols : Nstreams;
    if (TMPMEMNUM == 0)
        TMPMEMNUM = 1;

    if (size_t(n) > std::numeric_limits<size_t>::max() / sizeof(REAL)) {
        err << "Temporary matrix size overflows size_t." << endl;
        cleanup();
        return;
    }
    // hcMemGetInfo is absent from the supplied guide, so probe allocations and
    // halve the batch on failure. This preserves bounded memory use without
    // relying on an undocumented free-memory query.
    size_t tmp_bytes = 0;
    hcError_t tmp_alloc_status = hcSuccess;
    while (TMPMEMNUM > 0) {
        const size_t bytes_per_column = size_t(n) * sizeof(REAL);
        if (size_t(TMPMEMNUM) > std::numeric_limits<size_t>::max() / bytes_per_column) {
            err << "Temporary matrix size overflows size_t." << endl;
            cleanup();
            return;
        }
        tmp_bytes = size_t(TMPMEMNUM) * bytes_per_column;
        tmpMem = nullptr;
        tmp_alloc_status = hcMalloc((void**)&tmpMem, tmp_bytes);
        if (tmp_alloc_status == hcSuccess)
            break;
        TMPMEMNUM /= 2;
    }
    if (tmp_alloc_status != hcSuccess || tmpMem == nullptr) {
        hcCheck(tmp_alloc_status, "hcMalloc(tmpMem)");
        if (tmp_alloc_status == hcSuccess)
            err << "hcMalloc(tmpMem) returned a null pointer." << endl;
        tmpMem = nullptr;
        cleanup();
        return;
    }
    // hcMalloc does not zero memory; clear explicitly to preserve CUDA semantics.
    HC_RETURN_ON_ERR(hcMemset(tmpMem, 0, tmp_bytes), "hcMemset(tmpMem)");
    // hcMemset uses the default stream. Wait before the first kernels run on
    // independently created streams; the guide guarantees only same-stream order.
    HC_RETURN_ON_ERR(hcDeviceSynchronize(), "hcDeviceSynchronize(tmpMem init)");

    // calculate 1-norm of A and perturbation value for perturbation
    REAL pert = 0;
    if (PERTURB)
    {
        REAL norm_A = 0;
        for (unsigned i = 0; i < n; ++i)
        {
            REAL tmp = 0;
            for (unsigned j = A_sym.sym_c_ptr[i]; j < A_sym.sym_c_ptr[i+1]; ++j)
                tmp += fabs(A_sym.val[j]);
            if (norm_A < tmp)
                norm_A = tmp;
        }
        pert = (REAL)3.45e-4 * norm_A;
        out << "Gaussian elimination with static pivoting (GESP)..." << endl;
        out << "1-Norm of A matrix is " << norm_A << ", Perturbation value is " << pert << endl;
    }

    auto launch_batched_level = [&](unsigned level_head, int level_size, unsigned waves_per_block) {
        // HTHPCC wave width is 64; preserve the requested number of waves.
        const unsigned supported_waves = static_cast<unsigned>(maxWaveAlignedThreads / 64);
        if (waves_per_block > supported_waves)
            waves_per_block = supported_waves;
        dim3 dimBlock(waves_per_block * 64, 1);
        size_t mem_size = 0; // The pivot broadcast no longer uses dynamic shared memory.

        int remaining = level_size;
        unsigned chunk_idx = 0;
        while (remaining > 0) {
            unsigned rest_col = static_cast<unsigned>(remaining) > TMPMEMNUM ?
                TMPMEMNUM : static_cast<unsigned>(remaining);
            dim3 dimGrid(rest_col, 1);
            int in_level_pos = static_cast<int>(chunk_idx * TMPMEMNUM);
            if (!PERTURB)
                RL<<<dimGrid, dimBlock, mem_size>>>(sym_c_ptr_dev,
                                                    sym_r_idx_dev,
                                                    val_dev,
                                                    l_col_ptr_dev,
                                                    csr_r_ptr_dev,
                                                    csr_c_idx_dev,
                                                    csr_diag_ptr_dev,
                                                    level_idx_dev,
                                                    tmpMem,
                                                    n,
                                                    level_head,
                                                    in_level_pos);
            else
                RL_perturb<<<dimGrid, dimBlock, mem_size>>>(sym_c_ptr_dev,
                                                            sym_r_idx_dev,
                                                            val_dev,
                                                            l_col_ptr_dev,
                                                            csr_r_ptr_dev,
                                                            csr_c_idx_dev,
                                                            csr_diag_ptr_dev,
                                                            level_idx_dev,
                                                            tmpMem,
                                                            n,
                                                            level_head,
                                                            in_level_pos,
                                                            pert);
            remaining -= static_cast<int>(rest_col);
            ++chunk_idx;
        }
    };

    for (unsigned i = 0; i < num_lev; ++i)
    {
        int lev_size = A_sym.level_ptr[i + 1] - A_sym.level_ptr[i];
        if (lev_size <= 0)
            continue;

        // These inherited thresholds are CUDA-era tuning constants; tune on target hardware.
        if (lev_size > 896) {
            // Values count waves; block size below multiplies by HTHPCC's 64 lanes/wave.
            launch_batched_level(A_sym.level_ptr[i], lev_size, 2);
        }
        else if (lev_size > 448) {
            launch_batched_level(A_sym.level_ptr[i], lev_size, 4);
        }
        else if (lev_size > Nstreams) {
            launch_batched_level(A_sym.level_ptr[i], lev_size, 32);
        }
        else {
            // Small levels are mapped to one stream per column to reduce launch overhead.
            const int active_streams = (TMPMEMNUM < Nstreams)
                ? static_cast<int>(TMPMEMNUM) : Nstreams;
            for (int offset = 0; offset < lev_size; offset += active_streams) {
                for (int j = 0; j < active_streams; j++) {
                    if (j + offset < lev_size) {
                        const unsigned currentCol = A_sym.level_idx[A_sym.level_ptr[i] + j + offset];
                        const unsigned subMatSize = A_sym.csr_r_ptr[currentCol + 1]
                            - A_sym.csr_diag_ptr[currentCol] - 1;

                        if (!PERTURB)
                            RL_onecol_factorizeCurrentCol<<<1, onecolBlockThreads, 0, streams[j]>>>(sym_c_ptr_dev,
                                                                                        sym_r_idx_dev,
                                                                                        val_dev,
                                                                                        l_col_ptr_dev,
                                                                                        currentCol,
                                                                                        tmpMem,
                                                                                        j,
                                                                                        n);
                        else
                            RL_onecol_factorizeCurrentCol_perturb<<<1, onecolBlockThreads, 0, streams[j]>>>(sym_c_ptr_dev,
                                                                                                sym_r_idx_dev,
                                                                                                val_dev,
                                                                                                l_col_ptr_dev,
                                                                                                currentCol,
                                                                                                tmpMem,
                                                                                                j,
                                                                                                n,
                                                                                                pert);
                        if (subMatSize > 0)
                            RL_onecol_updateSubmat<<<subMatSize, onecolBlockThreads, 0, streams[j]>>>(sym_c_ptr_dev,
                                                                                          sym_r_idx_dev,
                                                                                          val_dev,
                                                                                          csr_c_idx_dev,
                                                                                          csr_diag_ptr_dev,
                                                                                          currentCol,
                                                                                          tmpMem,
                                                                                          j,
                                                                                          n);
                        RL_onecol_cleartmpMem<<<1, onecolBlockThreads, 0, streams[j]>>>(sym_c_ptr_dev,
                                                                           sym_r_idx_dev,
                                                                           l_col_ptr_dev,
                                                                           currentCol,
                                                                           tmpMem,
                                                                           j,
                                                                           n);
                    }
                }
            }
        }
        HC_RETURN_ON_ERR(hcGetLastError(), "kernel launch");
        HC_RETURN_ON_ERR(hcDeviceSynchronize(), "hcDeviceSynchronize");
    }

    HC_RETURN_ON_ERR(hcMemcpy(&(A_sym.val[0]), val_dev, nnz * sizeof(REAL), hcMemcpyDeviceToHost),
        "hcMemcpy(A_sym.val)");
    HC_RETURN_ON_ERR(hcEventRecord(stop), "hcEventRecord(stop)");
    HC_RETURN_ON_ERR(hcEventSynchronize(stop), "hcEventSynchronize(stop)");
    HC_RETURN_ON_ERR(hcEventElapsedTime(&time, start, stop), "hcEventElapsedTime");

    out << "Total GPU time: " << time << " ms" << endl;

#ifdef GLU_DEBUG
    // check NaN elements
    unsigned err_find = 0;
    for(unsigned i = 0; i < nnz; i++)
        if(isnan(A_sym.val[i]) || isinf(A_sym.val[i]))
            err_find++;

    if (err_find != 0)
        err << "LU data check: NaN/Inf found." << endl;
#endif
    cleanup();
#undef HC_RETURN_ON_ERR
}
