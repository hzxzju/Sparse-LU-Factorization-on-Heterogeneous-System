// Runs the production SF scheduler/helpers with one host lane per worker.
// This tests queue/checkpoint/waiter interleavings and exact set semantics;
// Additional host lane/barrier tests cover collective bitmap scans and export;
// these cannot validate device memory scope or hardware waves.
#include "symbolic.h"
#include <algorithm>
#include <atomic>
#include <condition_variable>
#include <memory>
#include <mutex>
#include <iostream>
#include <numeric>
#include <random>
#include <sstream>
#include <stdexcept>
#include <thread>
#include <vector>

#define GLU_SYMBOLIC_SF_HOST_TEST 1
#define __device__
#define __global__
#define __shared__
class Barrier {
    std::mutex mutex;
    std::condition_variable cv;
    unsigned lanes, arrived = 0, epoch = 0;
public:
    explicit Barrier(unsigned count) : lanes(count) {}
    void wait() {
        std::unique_lock<std::mutex> lock(mutex);
        unsigned current = epoch;
        if (++arrived == lanes) { arrived = 0; ++epoch; cv.notify_all(); }
        else cv.wait(lock, [&] { return epoch != current; });
    }
};
thread_local Barrier *active_barrier = nullptr;
void host_sync() { if (active_barrier) active_barrier->wait(); }
#define __sync_threads() host_sync()
struct Index { unsigned x = 0; };
thread_local Index threadIdx, blockIdx, blockDim, gridDim;
#include "../src/symbolic_sf.cuh"

using namespace glu_sym_sf;
void require(bool ok, const char *why) { if (!ok) throw std::runtime_error(why); }
struct Matrix { unsigned n; std::vector<unsigned> ai, ap; };
struct Run {
    Context c{};
    std::vector<std::vector<unsigned>> buffers;
    std::vector<Stats> stats;
    Run(const Matrix &a, unsigned workers, unsigned pool, unsigned chunks = 0) : buffers(23), stats(workers) {
        c.n = a.n; c.words = (a.n + 31) / 32; c.workers = workers;
        c.pool_capacity = pool; c.chunk_capacity = chunks ? chunks : pool;
        c.queue_capacity = 1;
        while (c.queue_capacity < a.n + workers + 1) c.queue_capacity *= 2;
        c.ap = a.ap.data(); c.ai = a.ai.data();
        unsigned index = 0;
        auto alloc = [&](unsigned count) -> unsigned * {
            auto &v = buffers[index++]; v.assign(size_t(count) + 2, 0xdeadbeef);
            return v.data() + 1;
        };
        c.sym_done = alloc(a.n); c.sym_state = alloc(a.n); c.cursor = alloc(a.n);
        c.wait_head = alloc(a.n); c.wait_next = alloc(a.n); c.checkpoint_head = alloc(a.n);
        c.chunk_start = alloc(c.chunk_capacity); c.chunk_len = alloc(c.chunk_capacity); c.chunk_next = alloc(c.chunk_capacity);
        c.checkpoint_pool = alloc(pool); c.final_pool = alloc(pool);
        c.final_start = alloc(a.n); c.col_len = alloc(a.n); c.diag_rel = alloc(a.n);
        c.work_bitmap = alloc(workers * c.words); c.dirty_bitmap = alloc(workers * c.words);
        c.queue_value = alloc(c.queue_capacity); c.queue_sequence = alloc(c.queue_capacity);
        c.counters = alloc(COUNTERS); c.stats = stats.data();
        threadIdx.x = 0; blockDim.x = 1;
        for (unsigned i = 0; i < std::max(c.queue_capacity, unsigned(COUNTERS)); ++i) {
            blockIdx.x = i; Initialize(c);
        }
    }
    unsigned execute(bool reverse, unsigned lanes = 1) {
        if (reverse) std::reverse(c.queue_value, c.queue_value + c.n);
        std::vector<Scratch> scratch(c.workers);
        std::vector<std::unique_ptr<Barrier>> barriers;
        for (unsigned w = 0; w < c.workers; ++w) barriers.emplace_back(new Barrier(lanes));
        std::vector<std::thread> threads;
        for (unsigned w = 0; w < c.workers; ++w) for (unsigned lane = 0; lane < lanes; ++lane)
            threads.emplace_back([&, w, lane] {
                threadIdx.x = lane; blockIdx.x = w; blockDim.x = lanes; gridDim.x = c.workers;
                if (lanes > 1) active_barrier = barriers[w].get();
                if ((w & 1) == 0) std::this_thread::yield();
                Worker(c, scratch[w]);
            });
        for (auto &thread : threads) thread.join();
        for (const auto &v : buffers) if (!v.empty())
            require(v.front() == 0xdeadbeef && v.back() == 0xdeadbeef, "arena out-of-bounds write");
        return c.counters[ERROR];
    }
    void compare(Symbolic_Matrix &gold) {
        require(c.counters[COMPLETED] == c.n, "lost wakeup / incomplete columns");
        unsigned nnz = 0;
        unsigned long long merges = 0, rows_read = 0, parks = 0, wakes = 0, input_rows = 0;
        for (unsigned col = 0; col < c.n; ++col) {
            require(c.sym_state[col] == FINISHED && c.sym_done[col] == 0, "incorrect done/state semantics");
            require(c.col_len[col] == gold.sym_c_ptr[col + 1] - gold.sym_c_ptr[col], "column length mismatch");
            require(c.diag_rel[col] == gold.l_col_ptr[col] - gold.sym_c_ptr[col], "diagonal offset mismatch");
            for (unsigned k = 0; k < c.col_len[col]; ++k) {
                unsigned row = c.final_pool[c.final_start[col] + k];
                require(row == gold.sym_r_idx[gold.sym_c_ptr[col] + k], "CPU/SF pattern mismatch");
                if (row < col) { ++merges; rows_read += gold.sym_c_ptr[row + 1] - gold.l_col_ptr[row]; }
            }
            // Checkpoint rows are delta-only, with no repeated snapshots.
            std::vector<bool> seen(c.n);
            for (unsigned h = c.checkpoint_head[col]; h != EMPTY; h = c.chunk_next[h]) {
                require(c.chunk_len[h] != 0, "empty checkpoint chunk");
                for (unsigned k = 0; k < c.chunk_len[h]; ++k) {
                    unsigned row = c.checkpoint_pool[c.chunk_start[h] + k];
                    require(row < c.n && !seen[row], "duplicate checkpoint row"); seen[row] = true;
                }
            }
            nnz += c.col_len[col];
        }
        unsigned long long actual_merges = 0, actual_reads = 0;
        for (const auto &s : stats) {
            actual_merges += s.merges; actual_reads += s.predecessor_rows;
            parks += s.parks; wakes += s.wakeups; input_rows += s.input_rows;
        }
        require(actual_merges == merges && actual_reads == rows_read, "predecessor re-read after checkpoint");
        require(parks == wakes, "park/wakeup mismatch");
        require(input_rows == c.ap[c.n], "input CSC was read again after resume");
        require(nnz == gold.nnz && nnz == c.counters[FINAL_USED], "nnz mismatch");
        require(c.counters[CHECKPOINT_USED] <= nnz, "checkpoint exceeds actual structural nnz");
        threadIdx.x = 0; blockDim.x = 1; gridDim.x = c.workers;
        std::vector<unsigned> ptr(size_t(c.n) + 1, 0), packed(nnz);
        for (unsigned col = 0; col < c.n; ++col) ptr[col + 1] = ptr[col] + c.col_len[col];
        for (unsigned w = 0; w < c.workers; ++w) { blockIdx.x = w; Pack(c, ptr.data(), packed.data()); }
        require(ptr == gold.sym_c_ptr && packed == gold.sym_r_idx, "packed CSC mismatch");
    }
};
Matrix random_matrix(unsigned n, unsigned seed) {
    Matrix a{n, {}, {0}};
    std::mt19937 rng(seed);
    for (unsigned col = 0; col < n; ++col) {
        for (unsigned r = n; r-- > 0;) if (rng() % 9 == 0) a.ai.push_back(r);
        a.ap.push_back(static_cast<unsigned>(a.ai.size()));
    }
    return a;
}
void check(Matrix a, unsigned workers, bool reverse) {
    std::ostringstream out, err;
    Symbolic_Matrix gold(a.n, out, err); gold.fill_in(a.ai.data(), a.ap.data());
    Run run(a, workers, std::max(1u, gold.nnz));
    require(run.execute(reverse) == OK, "unexpected scheduler error");
    run.compare(gold);
}
int main() {
    try {
        // Dynamic predecessors: column 4 starts at {0,4}, then discovers 1,3.
        Matrix dynamic{5, {1, 3, 2, 0, 4}, {0, 1, 2, 2, 3, 5}};
        for (unsigned w : {1u, 2u, 8u, 64u}) { check(dynamic, w, false); check(dynamic, w, true); }
        for (unsigned n : {0u, 1u, 31u, 32u, 33u, 65u, 257u}) {
            check(Matrix{n, {}, std::vector<unsigned>(size_t(n) + 1, 0)}, 8, true);
            check(random_matrix(n, 171), 8, true);
        }
        check(Matrix{8193, {}, std::vector<unsigned>(8194, 0)}, 2, true);
        // Over 1,000 concurrent full scheduler executions, including a grid
        // larger than the task count. Reverse issuance guarantees park/resume
        // even with a single resident worker; repeated rings wrap many times.
        for (unsigned repeat = 0; repeat < 1000; ++repeat) {
            unsigned workers[] = {1, 2, 8, 32};
            check(random_matrix(9 + repeat % 25, repeat * 17 + 3), workers[repeat % 4], repeat % 3 != 0);
        }
        for (unsigned lanes : {64u, 128u, 256u}) {
            Matrix a = random_matrix(9, lanes);
            std::ostringstream out, err;
            Symbolic_Matrix gold(a.n, out, err); gold.fill_in(a.ai.data(), a.ap.data());
            Run collective(a, lanes == 64 ? 2 : 1, gold.nnz);
            require(collective.execute(true, lanes) == OK, "collective worker failed");
            collective.compare(gold);
        }
        std::cout << "PASS collective bitmap/scan/export barriers with 64/128/256 host lanes\n";
        std::cout << "PASS 1000 SF scheduler runs: CPU patterns, delta checkpoints, exact predecessor reads, queue/waiters\n";
        Run final_full(dynamic, 4, 1);
        require(final_full.execute(true) != OK, "tiny pool did not fail");
        Matrix fanout{64, {}, {0}};
        for (unsigned col = 0; col < fanout.n; ++col) { fanout.ai.push_back(0); fanout.ap.push_back(col + 1); }
        Run chunks_full(fanout, 1, 4096, 1);
        require(chunks_full.execute(true) == CHUNKS_FULL, "chunk overflow did not fail safely");
        Run ticket(dynamic, 1, 64);
        store(ticket.c.counters + Q_HEAD, unsigned(INT_MAX) - ticket.c.queue_capacity);
        require(ticket.execute(false) == TICKET_LIMIT, "ticket overflow did not stop");
        Run queue_full(dynamic, 1, 64);
        store(queue_full.c.queue_sequence + (load(queue_full.c.counters + Q_TAIL) & (queue_full.c.queue_capacity - 1)), 0);
        require(!enqueue(queue_full.c, 0) && load(queue_full.c.counters + ERROR) == QUEUE_FULL, "queue overflow did not stop");
        std::cout << "PASS bounded arena/chunk/queue/ticket errors and safe worker exit (host simulation)\n";
    } catch (const std::exception &e) { std::cerr << e.what() << '\n'; return 1; }
}
