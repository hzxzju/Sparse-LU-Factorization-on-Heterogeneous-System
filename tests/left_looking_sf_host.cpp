// Run the production LL-SF kernel with one simulated CTA. Host barriers and
// acquire/release operations check arithmetic and collective control flow;
// they do not validate device memory scope, native waves or GPU performance.
#include "symbolic.h"
#include <algorithm>
#include <cmath>
#include <condition_variable>
#include <iostream>
#include <mutex>
#include <random>
#include <sstream>
#include <stdexcept>
#include <thread>
#include <vector>

class Barrier {
    std::mutex mutex;
    std::condition_variable cv;
    unsigned lanes, arrived = 0, epoch = 0;
public:
    explicit Barrier(unsigned count) : lanes(count) {}
    void wait() {
        std::unique_lock<std::mutex> lock(mutex);
        const unsigned previous = epoch;
        if (++arrived == lanes) { arrived = 0; ++epoch; cv.notify_all(); }
        else cv.wait(lock, [&] { return epoch != previous; });
    }
};

struct Index { unsigned x = 0; };
thread_local Index threadIdx, blockIdx, blockDim;
thread_local Barrier *active_barrier = nullptr;
#define __device__
#define __global__
// There is exactly one CTA per invocation, so static storage models shared
// storage. All host threads join before another invocation starts.
#define __shared__ static
#define __sync_threads() active_barrier->wait()

// CPU stand-ins for the device-only SDK helpers in sf_sync.cuh.
#define GLU_SF_SYNC_CUH
namespace glu_sf {
inline unsigned fetch_next_column(unsigned *next) {
    return __atomic_fetch_add(next, 1u, __ATOMIC_RELAXED);
}
inline unsigned load_done_acquire(unsigned *done) {
    return __atomic_load_n(done, __ATOMIC_ACQUIRE);
}
inline void wait_column_ready(unsigned *done) {
    if (threadIdx.x == 0)
        while (load_done_acquire(done) == 0u) {}
    __sync_threads();
}
inline void publish_column_done(unsigned *done) {
    __sync_threads();
    if (threadIdx.x == 0) __atomic_store_n(done, 1u, __ATOMIC_RELEASE);
    __sync_threads();
}
}
#include "../src/left_looking_sf.cuh"

static void run_case(const char *name, const std::vector<REAL> &a, unsigned n,
                     unsigned lanes, bool perturb, bool wave)
{
    std::vector<unsigned> rows, ptr(1, 0);
    std::vector<double> values;
    REAL norm = 0;
    for (unsigned j = 0; j < n; ++j) {
        REAL sum = 0;
        for (unsigned i = 0; i < n; ++i) {
            sum += std::fabs(a[i*n+j]);
            if (a[i*n+j] != 0) {
                rows.push_back(i);
                values.push_back(a[i*n+j]);
            }
        }
        norm = std::max(norm, sum);
        ptr.push_back(rows.size());
    }
    const REAL threshold = 3.45e-4 * norm;
    // Independent, right-looking dense elimination reference.
    std::vector<REAL> reference = a;
    for (unsigned k = 0; k < n; ++k) {
        if (perturb && std::fabs(reference[k*n+k]) < threshold)
            reference[k*n+k] = threshold;
        for (unsigned i = k + 1; i < n; ++i) {
            reference[i*n+k] /= reference[k*n+k];
            for (unsigned j = k + 1; j < n; ++j)
                reference[i*n+j] -= reference[i*n+k] * reference[k*n+j];
        }
    }
    std::ostringstream out, err;
    Symbolic_Matrix sym(n, out, err);
    sym.fill_in(rows.data(), ptr.data());
    sym.predictLU(rows.data(), ptr.data(), values.data());
    if (!err.str().empty()) throw std::runtime_error(err.str());

    // Repeat with reset task flags and dirty workspace, exercising reuse.
    const std::vector<REAL> original = sym.val;
    for (unsigned repeat = 0; repeat < 2; ++repeat) {
        sym.val = original;
        std::vector<REAL> workspace(n + 2, -12345);
        std::vector<unsigned> done(n, 0);
        unsigned next = 0;
        Barrier barrier(lanes);
        std::vector<std::thread> threads;
        for (unsigned lane = 0; lane < lanes; ++lane) {
            threads.emplace_back([&, lane] {
                threadIdx.x = lane;
                blockIdx.x = 0;
                blockDim.x = lanes;
                active_barrier = &barrier;
                if (wave)
                    LL_factorizeSyncFree<true>(sym.sym_c_ptr.data(), sym.sym_r_idx.data(),
                        sym.val.data(), sym.l_col_ptr.data(), workspace.data() + 1,
                        &next, done.data(), n, perturb, threshold);
                else
                    LL_factorizeSyncFree<>(sym.sym_c_ptr.data(), sym.sym_r_idx.data(),
                        sym.val.data(), sym.l_col_ptr.data(), workspace.data() + 1,
                        &next, done.data(), n, perturb, threshold);
            });
        }
        for (auto &thread : threads) thread.join();
        if (next != n + 1 || workspace.front() != -12345 || workspace.back() != -12345)
            throw std::runtime_error("task count / workspace bounds");
        for (unsigned j = 0; j < n; ++j) {
            if (done[j] != 1) throw std::runtime_error("unpublished column");
            for (unsigned p = sym.sym_c_ptr[j]; p < sym.sym_c_ptr[j+1]; ++p) {
                const REAL expected = reference[sym.sym_r_idx[p]*n+j];
                if (!std::isfinite(sym.val[p]) ||
                    std::fabs(sym.val[p] - expected) > 1e-10 * (1 + std::fabs(expected))) {
                    std::ostringstream detail;
                    detail << name << ": lanes=" << lanes << ", factor ("
                           << sym.sym_r_idx[p] << ',' << j << ")=" << sym.val[p]
                           << ", reference=" << expected;
                    throw std::runtime_error(detail.str());
                }
            }
        }
    }
    std::cout << "PASS " << name << ", update=" << (wave ? "wave" : "block")
              << ", block threads=" << lanes << std::endl;
}

int main()
{
    try {
        // A dense 66-column case exercises stripe lengths around 24, 48 and 64,
        // including offsets 24..63 assigned to the 24 active lanes and a second
        // element in a stride-64 stripe. It also covers all wave IDs/batches.
        const unsigned n = 66;
        std::vector<REAL> dense(n*n, 0);
        std::mt19937 rng(20261106);
        for (unsigned i = 0; i < n; ++i) {
            REAL sum = 0;
            for (unsigned j = 0; j < n; ++j) if (i != j) {
                const REAL value = (static_cast<int>(rng()%101) - 50) / 128.0;
                dense[i*n+j] = value;
                sum += std::fabs(value);
            }
            dense[i*n+i] = 1 + sum;
        }
        for (bool wave : {false, true})
        for (unsigned lanes : {64u, 128u, 256u}) {
            run_case("dense-search-boundaries", dense, n, lanes, false, wave);
            run_case("sparse-fill-in", {4,0,2,0,1,3,0,1,0,2,5,0,1,0,1,4}, 4, lanes, false, wave);
            run_case("empty-L", {2,1,2,0,3,1,0,0,4}, 3, lanes, false, wave);
            run_case("perturbed-pivot", {1,1,2,1,1,2,2,2,5}, 3, lanes, true, wave);
        }
    } catch (const std::exception &e) {
        std::cerr << e.what() << '\n';
        return 1;
    }
}
