#include <hc_runtime.h>
#include "symbolic.h"
#include "gpu_device.h"
#include "symbolic_sf.cuh"
#include <chrono>

#include <algorithm>
#include <climits>
#include <limits>
#include <sstream>

namespace {

void Check(hcError_t status, const char *operation)
{
    if (status != hcSuccess) {
        std::ostringstream message;
        message << operation << " failed (HTHPCC status " << static_cast<int>(status) << ": " << hcGetErrorString(status) << ')';
        throw std::runtime_error(message.str());
    }
}

template <typename T> class DeviceArray
{
public:
    T *data = nullptr;
    explicit DeviceArray(size_t count = 0) { allocate(count); }
    ~DeviceArray() { if (data) hcFree(data); }
    DeviceArray(const DeviceArray &) = delete;
    DeviceArray &operator=(const DeviceArray &) = delete;

    void allocate(size_t count)
    {
        if (!count) return;
        if (count > std::numeric_limits<size_t>::max() / sizeof(T))
            throw std::runtime_error("GPU symbolic allocation size overflow");
        Check(hcMalloc(reinterpret_cast<void **>(&data), count * sizeof(T)), "hcMalloc(symbolic)");
    }
    void upload(const T *source, size_t count)
    {
        if (count) Check(hcMemcpy(data, source, count * sizeof(T), hcMemcpyHostToDevice),
                         "hcMemcpy(symbolic H2D)");
    }
    void download(T *target, size_t count) const
    {
        if (count) Check(hcMemcpy(target, data, count * sizeof(T), hcMemcpyDeviceToHost),
                         "hcMemcpy(symbolic D2H)");
    }
};

// Declare after buffers used by a launch. On exceptions, wait before their
// destructors free memory; do not assume hcFree implicitly joins device work.
class DeviceWork
{
    bool pending = true;
public:
    ~DeviceWork() { if (pending) hcDeviceSynchronize(); }
    void finish(const char *operation)
    {
        Check(hcDeviceSynchronize(), operation);
        pending = false;
    }
};

// The input remains untouched: no sorting of ai without also sorting ax.
// Search accepts unsorted columns; duplicate coordinates are rejected explicitly.
void ValidateInput(unsigned n, const unsigned *ai, const unsigned *ap)
{
    if (n > static_cast<unsigned>(INT_MAX))
        throw std::invalid_argument("symbolic order exceeds the signed indices used by numeric LU");
    if (!ap || ap[0] != 0)
        throw std::invalid_argument("symbolic CSC requires ap[0] == 0");
    for (unsigned j = 0; j < n; ++j)
        if (ap[j] > ap[j + 1])
            throw std::invalid_argument("symbolic CSC column pointers must be monotone");
    if (ap[n] && !ai)
        throw std::invalid_argument("symbolic CSC row indices are missing");
    std::vector<unsigned> seen(n, UINT_MAX);
    for (unsigned j = 0; j < n; ++j) {
        for (unsigned p = ap[j]; p < ap[j + 1]; ++p) {
            const unsigned row = ai[p];
            if (row >= n)
                throw std::invalid_argument("symbolic CSC row index is out of range");
            if (seen[row] == j)
                throw std::invalid_argument("symbolic CSC contains a duplicate coordinate");
            seen[row] = j;
        }
    }
}

struct SearchState { unsigned current_size, next_size, job_cursor, epoch; };

// One CTA owns four n-element arrays. All threads enter every CTA barrier.
// Lane 0 claims a frontier job and broadcasts it with the SDK's 64-lane
// __shfl_sync (declared in __clang_hpcc_device_functions.h).
__device__ void SearchColumn(unsigned n, unsigned column, const unsigned *ap,
                             const unsigned *ai, unsigned *workspace,
                             SearchState &state, unsigned *error)
{
    const unsigned tid = threadIdx.x;
    unsigned *best = workspace;
    unsigned *current = best + n;
    unsigned *next = current + n;
    unsigned *epochs = next + n;
    for (unsigned r = tid; r < n; r += blockDim.x) {
        best[r] = UINT_MAX;
        epochs[r] = 0;
    }
    if (tid == 0) {
        state.current_size = 0;
        state.next_size = 0;
        state.job_cursor = 0;
        state.epoch = 1;
    }
    __sync_threads();
    if (tid == 0) best[column] = 0;
    __sync_threads();
    for (size_t p = size_t(ap[column]) + tid; p < ap[column + 1]; p += blockDim.x) {
        const unsigned r = ai[p];
        atomicMin(best + r, 0u);
        if (r < column) {
            const unsigned position = atomicAdd(&state.current_size, 1u);
            if (position < n) current[position] = r;
            else atomicExch(error, 1u);
        }
    }
    __sync_threads();

    const unsigned lane = tid % 64;
#if defined(GLU_TEST_CUDA_COMPAT)
    // Test-only CUDA has 32-lane warps, so use strided logical 64-lane jobs.
    // It cannot validate the native 64-lane shuffle or dynamic wave scheduling.
    const unsigned wave = tid / 64;
    const unsigned waves = blockDim.x / 64;
#endif
    while (state.current_size != 0) {
        unsigned job = 0;
#if defined(GLU_TEST_CUDA_COMPAT)
        job = wave;
#endif
        for (;;) {
#if !defined(GLU_TEST_CUDA_COMPAT)
            if (lane == 0) job = atomicAdd(&state.job_cursor, 1u);
            job = __shfl_sync(~0ULL, job, 0, 64);
#endif
            if (job >= state.current_size) break; // uniform within each wave
            const unsigned k = current[job];
            // Other waves may improve best[k] concurrently. Read atomically.
            const unsigned cost = atomicAdd(best + k, 0u);
            const unsigned candidate = cost > k + 1 ? cost : k + 1;
            for (size_t p = size_t(ap[k]) + lane; p < ap[k + 1]; p += 64) {
                const unsigned r = ai[p];
                const unsigned old = atomicMin(best + r, candidate);
                if (candidate < old && r < column &&
                    atomicExch(epochs + r, state.epoch) != state.epoch) {
                    const unsigned position = atomicAdd(&state.next_size, 1u);
                    if (position < n) next[position] = r;
                    else atomicExch(error, 1u);
                }
            }
#if defined(GLU_TEST_CUDA_COMPAT)
            job += waves;
#endif
        }
        // All frontier producers finish before any consumer reads their writes.
        __sync_threads();
        unsigned *temporary = current;
        current = next;
        next = temporary;
        if (tid == 0) {
            state.current_size = state.next_size <= n ? state.next_size : 0;
            state.next_size = 0;
            state.job_cursor = 0;
            ++state.epoch;
        }
        __sync_threads();
        if (state.epoch == 0) {
            for (unsigned r = tid; r < n; r += blockDim.x) epochs[r] = 0;
            __sync_threads();
            if (tid == 0) state.epoch = 1;
            __sync_threads();
        }
    }
}

__device__ bool ValidRow(unsigned r, unsigned column, const unsigned *best)
{
    return r == column || best[r] <= (r < column ? r : column);
}

__global__ void CountPatterns(unsigned n, unsigned first, const unsigned *ap,
                              const unsigned *ai, unsigned *workspace,
                              unsigned long long *counts, unsigned *diag_local,
                              unsigned *error)
{
    __shared__ SearchState state;
    __shared__ unsigned totals[256], diagonals[256];
    const unsigned tid = threadIdx.x;
    const unsigned column = first + blockIdx.x;
    unsigned *best = workspace + size_t(blockIdx.x) * 4 * n;
    SearchColumn(n, column, ap, ai, best, state, error);
    unsigned total = 0, diagonal = 0;
    for (unsigned r = tid; r < n; r += blockDim.x) {
        if (ValidRow(r, column, best)) {
            ++total;
            if (r < column) ++diagonal;
        }
    }
    totals[tid] = total;
    diagonals[tid] = diagonal;
    __sync_threads();
    for (unsigned stride = blockDim.x / 2; stride; stride /= 2) {
        if (tid < stride) {
            totals[tid] += totals[tid + stride];
            diagonals[tid] += diagonals[tid + stride];
        }
        __sync_threads();
    }
    if (tid == 0) {
        counts[column] = totals[0];
        diag_local[column] = diagonals[0];
    }
}

// Hierarchical 64-bit exclusive scan. Only narrow after checking total nnz.
__global__ void ScanTiles(const unsigned long long *input, unsigned long long *output,
                          unsigned long long *sums, unsigned n)
{
    __shared__ unsigned long long values[256];
    const unsigned tid = threadIdx.x;
    const unsigned i = blockIdx.x * blockDim.x + tid;
    const unsigned long long value = i < n ? input[i] : 0;
    values[tid] = value;
    __sync_threads();
    for (unsigned stride = 1; stride < blockDim.x; stride *= 2) {
        const unsigned long long add = tid >= stride ? values[tid - stride] : 0;
        __sync_threads();
        values[tid] += add;
        __sync_threads();
    }
    if (i < n) output[i] = values[tid] - value;
    if (i == n - 1) output[n] = values[tid];
    if (tid == blockDim.x - 1) sums[blockIdx.x] = values[tid];
}

__global__ void AddTileOffsets(unsigned long long *output,
                               const unsigned long long *offsets, unsigned n)
{
    const unsigned i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) output[i] += offsets[blockIdx.x];
    if (i == n - 1) output[n] += offsets[blockIdx.x];
}

void Scan(const unsigned long long *input, unsigned long long *output,
          unsigned n, unsigned threads)
{
    const unsigned blocks = (n - 1) / threads + 1;
    DeviceArray<unsigned long long> sums(blocks);
    DeviceArray<unsigned long long> offsets(blocks > 1 ? size_t(blocks) + 1 : 0);
    DeviceWork work;
    ScanTiles<<<blocks, threads>>>(input, output, sums.data, n);
    Check(hcGetLastError(), "ScanTiles launch");
    if (blocks > 1) {
        Scan(sums.data, offsets.data, blocks, threads);
        AddTileOffsets<<<blocks, threads>>>(output, offsets.data, n);
        Check(hcGetLastError(), "AddTileOffsets launch");
    }
    // Keep temporary buffers alive through their final device use.
    work.finish("symbolic scan synchronization");
}

__global__ void NarrowPointers(const unsigned long long *wide, unsigned *ptr,
                               unsigned *diagonal, const unsigned *diag_local, unsigned n)
{
    const unsigned i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i <= n) ptr[i] = static_cast<unsigned>(wide[i]);
    if (i < n) diagonal[i] = static_cast<unsigned>(wide[i]) + diag_local[i];
}

__global__ void WritePatterns(unsigned n, unsigned first, const unsigned *ap,
                              const unsigned *ai, unsigned *workspace,
                              const unsigned *ptr, unsigned *rows, unsigned *error)
{
    __shared__ SearchState state;
    __shared__ unsigned offsets[256];
    __shared__ unsigned written;
    const unsigned tid = threadIdx.x;
    const unsigned column = first + blockIdx.x;
    unsigned *best = workspace + size_t(blockIdx.x) * 4 * n;
    SearchColumn(n, column, ap, ai, best, state, error);
    if (tid == 0) written = 0;
    __sync_threads();
    // Scan contiguous row tiles to guarantee strictly increasing CSC rows.
    for (unsigned first_row = 0; first_row < n; first_row += blockDim.x) {
        const unsigned r = first_row + tid;
        const unsigned valid = r < n && ValidRow(r, column, best);
        offsets[tid] = valid;
        __sync_threads();
        for (unsigned stride = 1; stride < blockDim.x; stride *= 2) {
            const unsigned add = tid >= stride ? offsets[tid - stride] : 0;
            __sync_threads();
            offsets[tid] += add;
            __sync_threads();
        }
        if (valid) {
            const unsigned local = written + offsets[tid] - 1;
            if (local < ptr[column + 1] - ptr[column]) rows[ptr[column] + local] = r;
            else atomicExch(error, 2u);
        }
        __sync_threads();
        if (tid == 0) written += offsets[blockDim.x - 1];
        __sync_threads();
    }
    if (tid == 0 && written != ptr[column + 1] - ptr[column]) atomicExch(error, 2u);
}

struct HostPattern
{
    std::vector<unsigned> ptr, rows, diagonal;
    int device = -1;
};

void VerifyArray(const std::vector<unsigned> &gpu, const std::vector<unsigned> &cpu,
                 const char *name)
{
    if (gpu.size() != cpu.size()) {
        std::ostringstream message;
        message << "GPU/CPU " << name << " size mismatch: " << gpu.size()
                << " vs " << cpu.size();
        throw std::runtime_error(message.str());
    }
    for (size_t i = 0; i < gpu.size(); ++i) {
        if (gpu[i] == cpu[i]) continue;
        std::ostringstream message;
        message << "GPU/CPU " << name << '[' << i << "] mismatch: GPU="
                << gpu[i] << ", CPU=" << cpu[i];
        throw std::runtime_error(message.str());
    }
}

HostPattern BuildPattern(unsigned n, const unsigned *ai, const unsigned *ap,
                         const SymbolicConfig &config, int preferred, std::ostream &out)
{
    HostPattern result;
    result.ptr.resize(size_t(n) + 1, 0);
    result.diagonal.resize(n);
    if (!n) return result;

    hcDeviceProp_t properties;
    result.device = glu_gpu::SelectDevice(preferred,
        glu_gpu::DeviceRequirements(config.block_size, n / config.block_size + 1),
        properties, out);

    DeviceArray<unsigned> d_ap(size_t(n) + 1), d_ai(ap[n]), d_diag_local(n);
    DeviceArray<unsigned long long> d_counts(n), d_wide_ptr(size_t(n) + 1);
    DeviceArray<unsigned> d_ptr(size_t(n) + 1), d_diagonal(n), d_error(1);
    d_ap.upload(ap, size_t(n) + 1);
    d_ai.upload(ai, ap[n]);
    Check(hcMemset(d_error.data, 0, sizeof(unsigned)), "hcMemset(symbolic error)");

    size_t free_bytes = 0, total_bytes = 0;
    Check(hcMemGetInfo(&free_bytes, &total_bytes), "hcMemGetInfo");
    if (size_t(n) > std::numeric_limits<size_t>::max() / (4 * sizeof(unsigned)))
        throw std::overflow_error("GPU symbolic workspace size overflow");
    const size_t slot_bytes = size_t(n) * 4 * sizeof(unsigned);
    // Reserve half of currently free memory for the output and other users.
    unsigned slots = static_cast<unsigned>(std::min<size_t>(
        std::min(config.batch_cols, n), std::min<size_t>(
            properties.maxGridSize[0], (free_bytes / 2) / slot_bytes)));
    if (!slots) throw std::runtime_error("Insufficient memory for a GPU symbolic workspace slot");
    DeviceArray<unsigned> workspace;
    // The memory estimate can race other users; retry smaller batches safely.
    for (;;) {
        const hcError_t status = hcMalloc(reinterpret_cast<void **>(&workspace.data),
                                         size_t(slots) * slot_bytes);
        if (status == hcSuccess) break;
        workspace.data = nullptr;
        hcGetLastError();
        if (slots == 1) Check(status, "hcMalloc(symbolic workspace)");
        slots = std::max(1u, slots / 2);
    }
    out << "GPU symbolic batch columns: " << slots
        << ", block size: " << config.block_size << '\n';

    {
        DeviceWork count_work;
        for (unsigned first = 0; first < n;) {
            const unsigned batch = std::min(slots, n - first);
            CountPatterns<<<batch, config.block_size>>>(n, first, d_ap.data, d_ai.data,
                workspace.data, d_counts.data, d_diag_local.data, d_error.data);
            Check(hcGetLastError(), "CountPatterns launch");
            first += batch;
        }
        Scan(d_counts.data, d_wide_ptr.data, n, config.block_size);
        count_work.finish("GPU symbolic count synchronization");
    }
    unsigned error = 0;
    d_error.download(&error, 1);
    if (error) throw std::runtime_error("GPU symbolic frontier capacity exceeded");
    unsigned long long nnz = 0;
    Check(hcMemcpy(&nnz, d_wide_ptr.data + n, sizeof(nnz), hcMemcpyDeviceToHost),
          "hcMemcpy(symbolic nnz)");
    if (nnz > UINT_MAX)
        throw std::overflow_error("GPU symbolic nnz exceeds the unsigned CSC interface");
    DeviceArray<unsigned> d_rows(static_cast<size_t>(nnz));
    DeviceWork write_work;
    const unsigned blocks = n / config.block_size + 1;
    NarrowPointers<<<blocks, config.block_size>>>(d_wide_ptr.data, d_ptr.data,
        d_diagonal.data, d_diag_local.data, n);
    Check(hcGetLastError(), "NarrowPointers launch");
    for (unsigned first = 0; first < n;) {
        const unsigned batch = std::min(slots, n - first);
        WritePatterns<<<batch, config.block_size>>>(n, first, d_ap.data, d_ai.data,
            workspace.data, d_ptr.data, d_rows.data, d_error.data);
        Check(hcGetLastError(), "WritePatterns launch");
        first += batch;
    }
    write_work.finish("GPU symbolic synchronization");
    d_error.download(&error, 1);
    if (error) throw std::runtime_error("GPU symbolic queue overflow or count/write mismatch");
    result.rows.resize(static_cast<size_t>(nnz));
    d_ptr.download(result.ptr.data(), size_t(n) + 1);
    d_diagonal.download(result.diagonal.data(), n);
    d_rows.download(result.rows.data(), result.rows.size());
    return result;
}

// A failed arena attempt owns all buffers and completes its kernel before
// destruction. Growth always restarts the entire computation with new pointers.
using WallClock = std::chrono::steady_clock;
double Milliseconds(WallClock::time_point start) {
    return std::chrono::duration<double, std::milli>(WallClock::now() - start).count();
}
class WorkerEvents {
    hcEvent_t begin{}, end{};
    bool have_begin = false, have_end = false;
public:
    WorkerEvents() {
        Check(hcEventCreate(&begin), "hcEventCreate(symbolic start)");
        have_begin = true;
        const hcError_t status = hcEventCreate(&end);
        if (status != hcSuccess) { hcEventDestroy(begin); have_begin = false; Check(status, "hcEventCreate(symbolic end)"); }
        have_end = true;
    }
    ~WorkerEvents() { if (have_end) hcEventDestroy(end); if (have_begin) hcEventDestroy(begin); }
    void start() { Check(hcEventRecord(begin, 0), "hcEventRecord(symbolic start)"); }
    double stop() {
        Check(hcEventRecord(end, 0), "hcEventRecord(symbolic end)");
        Check(hcEventSynchronize(end), "hcEventSynchronize(symbolic)");
        float ms = 0;
        Check(hcEventElapsedTime(&ms, begin, end), "hcEventElapsedTime(symbolic)");
        return ms;
    }
};

struct SFMetrics {
    double compute = 0, pack = 0, h2d = 0, d2h = 0;
    size_t peak_bytes = 0;
    unsigned retries = 0, allocation_retries = 0;
    glu_sym_sf::Stats stats{};
};

void LogSFMetrics(const SFMetrics &metrics, size_t input_nnz, size_t output_nnz,
                  size_t workers, size_t bitmap_bytes, size_t pool, size_t chunks,
                  const unsigned *counters, std::ostream &out)
{
    using namespace glu_sym_sf;
    out << "symbolic_gpu_compute_ms: " << metrics.compute << '\n'
        << "symbolic_gpu_pack_ms: " << metrics.pack << '\n'
        << "symbolic_gpu_h2d_ms: " << metrics.h2d << '\n'
        << "symbolic_gpu_d2h_ms: " << metrics.d2h << '\n'
        << "symbolic_workers: " << workers << '\n'
        << "symbolic_input_nnz: " << input_nnz << '\n'
        << "symbolic_output_nnz: " << output_nnz << '\n'
        << "symbolic_fill_ratio: " << (input_nnz ? double(output_nnz) / input_nnz : 0.) << '\n'
        << "symbolic_bitmap_bytes: " << bitmap_bytes << '\n'
        << "symbolic_peak_workspace_bytes: " << metrics.peak_bytes << '\n'
        << "symbolic_checkpoint_pool_used: " << counters[CHECKPOINT_USED] << '\n'
        << "symbolic_checkpoint_pool_capacity: " << pool << '\n'
        << "symbolic_checkpoint_chunks_used: " << counters[CHUNKS_USED] << '\n'
        << "symbolic_checkpoint_chunk_capacity: " << chunks << '\n'
        << "symbolic_final_pool_used: " << counters[FINAL_USED] << '\n'
        << "symbolic_final_pool_capacity: " << pool << '\n'
        << "symbolic_task_parks: " << metrics.stats.parks << '\n'
        << "symbolic_task_wakeups: " << metrics.stats.wakeups << '\n'
        << "symbolic_pred_merges: " << metrics.stats.merges << '\n'
        << "symbolic_predecessor_rows_read: " << metrics.stats.predecessor_rows << '\n'
        << "symbolic_duplicate_rows_eliminated: " << metrics.stats.duplicates << '\n'
        << "symbolic_checkpoint_rows_restored: " << metrics.stats.restores << '\n'
        << "symbolic_input_rows_read: " << metrics.stats.input_rows << '\n'
        << "symbolic_empty_queue_polls: " << metrics.stats.empty_polls << '\n'
        << "symbolic_closed_wait_races: " << metrics.stats.closed_races << '\n'
        << "symbolic_oom_retries: " << metrics.retries + metrics.allocation_retries << '\n'
        << "symbolic_cpu_fallback_count: 0" << std::endl;
}

HostPattern BuildSFPattern(unsigned n, const unsigned *ai, const unsigned *ap,
                           const SymbolicConfig &config, int preferred, std::ostream &out)
{
    using namespace glu_sym_sf;
    HostPattern result;
    result.ptr.assign(size_t(n) + 1, 0);
    result.diagonal.resize(n);
    if (!n) {
        unsigned counters[COUNTERS] = {};
        LogSFMetrics(SFMetrics{}, 0, 0, 0, 0, 0, 0, counters, out);
        return result;
    }
    hcDeviceProp_t properties{};
    result.device = glu_gpu::SelectDevice(preferred,
        glu_gpu::DeviceRequirements(config.block_size, n / config.block_size + 1), properties, out);
    size_t free_bytes = 0, total_bytes = 0;
    Check(hcMemGetInfo(&free_bytes, &total_bytes), "hcMemGetInfo(symbolic SF)");
    const size_t words = (size_t(n) + 31) / 32;
    size_t workers = config.sf_workers ? config.sf_workers : 2 * size_t(std::max(1, properties.multiProcessorCount));
    workers = std::min(workers, std::min(size_t(n), size_t(properties.maxGridSize[0])));
    // Half the available memory is the complete SF device allocation budget,
    // including the eventual packed rows. This leaves room for other clients.
    const size_t budget = free_bytes / 2;
    SFMetrics metrics;
    size_t pool = config.pool_entries ? config.pool_entries :
        std::max(size_t(n), std::max(size_t(ap[n]) + n, size_t(ap[n]) * 4));
    pool = std::min(pool, size_t(UINT_MAX));
    if (config.max_pool_entries) pool = std::min(pool, size_t(config.max_pool_entries));
    static_assert(sizeof(size_t) >= 8 && sizeof(unsigned) == 4,
                  "Symbolic SF requires 64-bit host sizes and 32-bit CSC indices");
    // n<=INT_MAX, W<=n, pools<=UINT_MAX and queue<=2^29 bound every
    // allocation product/sum below 2^61, including both worker bitmaps.
    size_t requested_chunks = 0;
    bool reduce_allocation = false;
    for (;;) {
        size_t queue_capacity = 1;
        const size_t need = size_t(n) + workers + 1;
        while (queue_capacity < need && queue_capacity < (size_t(1) << 29)) queue_capacity *= 2;
        if (queue_capacity < need) throw std::runtime_error("GPU symbolic SF queue exceeds 32-bit ticket limit");
        if ((std::max(queue_capacity, size_t(COUNTERS)) + config.block_size - 1) / config.block_size >
            size_t(properties.maxGridSize[0]))
            throw std::runtime_error("GPU symbolic SF initialization grid exceeds device limits");
        // At most one metadata chunk per checkpoint row. Sparse initial chunk
        // budget avoids reserving three additional arrays of size final nnz.
        const size_t chunks = std::min(pool, std::max(requested_chunks, std::max(size_t(n), pool / 8)));
        const size_t bitmap_bytes = workers * words * 2 * sizeof(unsigned);
        const size_t base_bytes = (size_t(n) * 12 + 2 + ap[n] + 2 * queue_capacity + COUNTERS) * sizeof(unsigned)
            + workers * sizeof(Stats) + bitmap_bytes + (size_t(n) * 2 + 1) * sizeof(unsigned long long);
        // final + checkpoint + packed rows + chunk metadata, plus the recursive
        // scan temporaries (bounded by 2*n 64-bit elements for >=64 lanes).
        const size_t peak_bytes = base_bytes + pool * 3 * sizeof(unsigned) + chunks * 3 * sizeof(unsigned)
            + size_t(n) * 2 * sizeof(unsigned long long);
        if (peak_bytes > budget || reduce_allocation) {
            if (workers > 1) { workers = std::max(size_t(1), workers / 2); reduce_allocation = false; continue; }
            if (pool > 1 && metrics.retries == 0) { pool = std::max(size_t(1), pool / 2); reduce_allocation = false; continue; }
            throw std::runtime_error("GPU symbolic SF memory budget/allocation exhausted");
        }
        metrics.peak_bytes = std::max(metrics.peak_bytes, peak_bytes);
        unsigned failure = OK;
        unsigned counters[COUNTERS] = {};
        try {
            DeviceArray<unsigned> d_ap(size_t(n) + 1), d_ai(ap[n]);
            DeviceArray<unsigned> done(n), state(n), cursor(n), wait_head(n), wait_next(n), checkpoint_head(n);
            DeviceArray<unsigned> chunk_start(chunks), chunk_len(chunks), chunk_next(chunks);
            DeviceArray<unsigned> checkpoint_pool(pool), final_pool(pool), final_start(n), col_len(n), diag_rel(n);
            DeviceArray<unsigned> work(workers * words), dirty(workers * words);
            DeviceArray<unsigned> q_value(queue_capacity), q_seq(queue_capacity), d_counters(COUNTERS);
            DeviceArray<Stats> d_stats(workers);
            DeviceArray<unsigned long long> counts(n), wide(size_t(n) + 1);
            DeviceArray<unsigned> ptr(size_t(n) + 1), diagonal(n);
            Context c{n, static_cast<unsigned>(words), static_cast<unsigned>(workers),
                static_cast<unsigned>(queue_capacity), static_cast<unsigned>(pool), static_cast<unsigned>(chunks),
                d_ap.data, d_ai.data, done.data, state.data, cursor.data, wait_head.data, wait_next.data,
                checkpoint_head.data, chunk_start.data, chunk_len.data, chunk_next.data,
                checkpoint_pool.data, final_pool.data, final_start.data, col_len.data, diag_rel.data,
                work.data, dirty.data, q_value.data, q_seq.data, d_counters.data, d_stats.data};
            WorkerEvents events;
            DeviceWork pending;
            auto stage = WallClock::now();
            d_ap.upload(ap, size_t(n) + 1); d_ai.upload(ai, ap[n]);
            Check(hcDeviceSynchronize(), "symbolic SF H2D synchronization");
            metrics.h2d += Milliseconds(stage);
            Initialize<<<static_cast<unsigned>((std::max(queue_capacity, size_t(COUNTERS)) + config.block_size - 1) / config.block_size), config.block_size>>>(c);
            Check(hcGetLastError(), "symbolic SF init launch");
            Check(hcDeviceSynchronize(), "symbolic SF init synchronization");
            out << "GPU symbolic SF workers: " << workers << ", block size: " << config.block_size
                << ", bitmap bytes: " << bitmap_bytes << ", pool capacity: " << pool
                << ", chunk capacity: " << chunks << std::endl;
            events.start();
            Factorize<<<static_cast<unsigned>(workers), config.block_size>>>(c);
            Check(hcGetLastError(), "symbolic SF worker launch");
            metrics.compute += events.stop();
            pending.finish("symbolic SF worker synchronization");
            stage = WallClock::now();
            d_counters.download(counters, COUNTERS);
            std::vector<Stats> stats(workers);
            d_stats.download(stats.data(), workers);
            for (const Stats &s : stats) {
                metrics.stats.parks += s.parks; metrics.stats.wakeups += s.wakeups;
                metrics.stats.merges += s.merges; metrics.stats.predecessor_rows += s.predecessor_rows;
                metrics.stats.duplicates += s.duplicates; metrics.stats.restores += s.restores;
                metrics.stats.empty_polls += s.empty_polls; metrics.stats.closed_races += s.closed_races;
                metrics.stats.input_rows += s.input_rows;
            }
            metrics.d2h += Milliseconds(stage);
            failure = counters[ERROR];
            if (failure == OK) {
                if (counters[COMPLETED] != n) throw std::runtime_error("GPU symbolic SF incomplete columns");
                stage = WallClock::now();
                DeviceArray<unsigned> rows(counters[FINAL_USED]);
                DeviceWork pack_work;
                const unsigned blocks = n / config.block_size + 1;
                WideCounts<<<blocks, config.block_size>>>(c, counts.data);
                Check(hcGetLastError(), "symbolic SF counts launch");
                Scan(counts.data, wide.data, n, config.block_size);
                unsigned long long nnz = 0;
                Check(hcMemcpy(&nnz, wide.data + n, sizeof(nnz), hcMemcpyDeviceToHost), "symbolic SF nnz D2H");
                if (nnz > UINT_MAX) throw std::overflow_error("GPU symbolic SF nnz exceeds unsigned CSC");
                if (nnz != counters[FINAL_USED]) throw std::runtime_error("GPU symbolic SF arena/scan mismatch");
                NarrowPointers<<<blocks, config.block_size>>>(wide.data, ptr.data, diagonal.data, diag_rel.data, n);
                Check(hcGetLastError(), "symbolic SF narrow launch");
                Pack<<<static_cast<unsigned>(workers), config.block_size>>>(c, ptr.data, rows.data);
                Check(hcGetLastError(), "symbolic SF pack launch");
                pack_work.finish("symbolic SF pack synchronization");
                metrics.pack += Milliseconds(stage);
                stage = WallClock::now();
                result.rows.resize(static_cast<size_t>(nnz));
                ptr.download(result.ptr.data(), size_t(n) + 1);
                diagonal.download(result.diagonal.data(), n);
                rows.download(result.rows.data(), result.rows.size());
                metrics.d2h += Milliseconds(stage);
            }
        } catch (const std::runtime_error &e) {
            // hcMalloc failures are retryable with fewer bitmaps / smaller
            // initial arenas. Other runtime failures retain their diagnosis.
            if (std::string(e.what()).find("hcMalloc(symbolic)") == std::string::npos) throw;
            hcGetLastError();
            reduce_allocation = true;
            ++metrics.allocation_retries;
            out << "GPU symbolic SF allocation retry: " << e.what() << std::endl;
            continue;
        }
        if (failure == OK) {
            LogSFMetrics(metrics, ap[n], result.rows.size(), workers, bitmap_bytes, pool, chunks, counters, out);
            return result;
        }
        if (failure != FINAL_FULL && failure != CHECKPOINT_FULL && failure != CHUNKS_FULL)
            throw std::runtime_error("GPU symbolic SF scheduler error " + std::to_string(failure));
        if (failure == CHUNKS_FULL && chunks < pool) {
            requested_chunks = std::min(pool, chunks * 2);
            ++metrics.retries;
            out << "GPU symbolic SF chunk retry: " << chunks << " -> " << requested_chunks << std::endl;
            continue;
        }
        if (pool == UINT_MAX && (failure == FINAL_FULL || failure == CHECKPOINT_FULL))
            throw std::overflow_error("GPU symbolic SF structure exceeds unsigned CSC capacity");
        size_t grown = std::min(size_t(UINT_MAX), pool * 2);
        if (config.max_pool_entries) grown = std::min(grown, size_t(config.max_pool_entries));
        if (grown <= pool) throw std::runtime_error("GPU symbolic SF pool limit reached (error " + std::to_string(failure) + ")");
        out << "GPU symbolic SF pool retry: error=" << failure << ", capacity=" << pool << " -> " << grown << std::endl;
        pool = grown;
        ++metrics.retries;
    }
}

void ValidatePattern(unsigned n, const HostPattern &p)
{
    if (p.ptr.size() != size_t(n) + 1 || p.ptr[0] != 0 || p.ptr.back() != p.rows.size() || p.diagonal.size() != n)
        throw std::runtime_error("GPU symbolic invalid CSC sizes");
    for (unsigned col = 0; col < n; ++col) {
        if (p.ptr[col] >= p.ptr[col + 1] || p.ptr[col + 1] > p.rows.size() ||
            p.diagonal[col] < p.ptr[col] || p.diagonal[col] >= p.ptr[col + 1] || p.rows[p.diagonal[col]] != col)
            throw std::runtime_error("GPU symbolic invalid structural diagonal");
        for (size_t k = p.ptr[col]; k < p.ptr[col + 1]; ++k)
            if (p.rows[k] >= n || (k > p.ptr[col] && p.rows[k - 1] >= p.rows[k]))
                throw std::runtime_error("GPU symbolic rows are not a canonical set");
    }
}

} // namespace

void Symbolic_Matrix::fill_in(unsigned *ai, unsigned *ap, const SymbolicConfig &config)
{
    const auto symbolic_start = WallClock::now();
    ValidateSymbolicConfiguration(config);
    // All configured backends share the canonical CSC contract. Keep the
    // historical two-argument CPU entry point available as the golden path.
    ValidateInput(n, ai, ap);
    SymbolicBackend backend = config.backend;
    if (backend == SymbolicBackend::Auto) {
        const auto auto_start = WallClock::now();
        const SymbolicAutoSelection selection = SelectAutomaticSymbolicBackend(n, ap);
        const double auto_ms = Milliseconds(auto_start);
        backend = selection.backend;
        m_out << "Symbolic requested backend: auto\n"
              << "symbolic_auto_input_nnz: " << selection.input_nnz << '\n'
              << "symbolic_auto_max_column_nnz: " << selection.max_column_nnz << '\n'
              << "symbolic_auto_copy_lower_bound: " << selection.copy_lower_bound << '\n'
              << "symbolic_auto_ms: " << auto_ms << '\n'
              << "Symbolic auto selected backend: " << SymbolicBackendName(backend) << std::endl;
    }
    if (backend == SymbolicBackend::CPU) {
        m_out << "Symbolic backend: cpu" << std::endl;
        const auto start = WallClock::now();
        fill_in(ai, ap);
        m_out << "symbolic_cpu_ms: " << Milliseconds(start) << std::endl;
        return;
    }
    if (config.backend != SymbolicBackend::Auto)
        m_out << "Symbolic requested backend: " << SymbolicBackendName(backend) << std::endl;
    double verification_ms = 0;
    HostPattern pattern;
    try {
        pattern = backend == SymbolicBackend::GPU_SF ?
            BuildSFPattern(n, ai, ap, config, gpu_device, m_out) :
            BuildPattern(n, ai, ap, config, gpu_device, m_out);
    } catch (const std::overflow_error &) {
        // The CPU backend has the same index limit, so fallback cannot help.
        throw;
    } catch (const std::runtime_error &error) {
        if (!config.fallback_to_cpu) throw;
        m_out << "symbolic_cpu_fallback_count: 1\nGPU symbolic fallback: " << error.what()
              << "; falling back to CPU.\nSymbolic backend: cpu (fallback)" << std::endl;
        m_out << "symbolic_gpu_failed_ms: " << Milliseconds(symbolic_start) << std::endl;
        const auto cpu_start = WallClock::now();
        fill_in(ai, ap);
        m_out << "symbolic_cpu_ms: " << Milliseconds(cpu_start) << std::endl;
        return;
    }
    ValidatePattern(n, pattern);
    // Verification failures always stop; they must never disappear in fallback.
    if (config.verify) {
        std::ostringstream reference_output, reference_errors;
        Symbolic_Matrix reference(n, reference_output, reference_errors);
        const auto cpu_start = WallClock::now();
        reference.fill_in(ai, ap);
        m_out << "symbolic_cpu_ms: " << Milliseconds(cpu_start) << std::endl;
        VerifyArray(pattern.ptr, reference.sym_c_ptr, "sym_c_ptr");
        VerifyArray(pattern.rows, reference.sym_r_idx, "sym_r_idx");
        VerifyArray(pattern.diagonal, reference.l_col_ptr, "l_col_ptr");
        m_out << "GPU symbolic verification: passed (all CSC arrays match CPU)" << std::endl;
        verification_ms = Milliseconds(cpu_start);
    }
    // Commit only a complete result. CPU consumers retain vector ownership;
    // no temporary device pointers escape the adapter.
    sym_c_ptr.swap(pattern.ptr);
    sym_r_idx.swap(pattern.rows);
    l_col_ptr.swap(pattern.diagonal);
    gpu_device = pattern.device;
    symbolic_backend = backend;
    nnz = sym_c_ptr.back();
    reset_derived_structure();
    m_out << "Symbolic backend: " << SymbolicBackendName(symbolic_backend)
          << " (fill-in; CSR/values/levels on CPU)\n"
          << "Symbolic nonzero: " << nnz << std::endl;
    if (backend == SymbolicBackend::GPU_SF)
        m_out << "symbolic_gpu_total_ms: " << Milliseconds(symbolic_start) - verification_ms << std::endl;
}
