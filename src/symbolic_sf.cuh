#ifndef GLU_SYMBOLIC_SF_CUH
#define GLU_SYMBOLIC_SF_CUH

// Structure only. Numeric's done=1 means completed; symbolic's sym_done=0
// means runnable. Completion is a separate, release-published FINISHED state.
#ifndef GLU_SYMBOLIC_SF_HOST_TEST
#include "sf_sync.cuh"
#endif
#include <climits>
#include <cstddef>

namespace glu_sym_sf {

enum State : unsigned { READY, RUNNING, WAITING, FINISHED };
enum Error : unsigned { OK, FINAL_FULL, CHECKPOINT_FULL, CHUNKS_FULL,
                        QUEUE_FULL, TICKET_LIMIT, BAD_STATE };
const unsigned EMPTY = UINT_MAX, CLOSED = UINT_MAX - 1;

// Device-scope ordered RMWs are essential for the waiter list and queue.
__device__ inline unsigned load(unsigned *p) {
#if defined(GLU_SYMBOLIC_SF_HOST_TEST)
    return __atomic_load_n(p, __ATOMIC_ACQUIRE);
#elif defined(GLU_TEST_CUDA_COMPAT)
    return glu_sf::load_done_acquire(p);
#else
    return __hpcc_atomic_load(p, __ATOMIC_ACQUIRE, memory_scope_device);
#endif
}
__device__ inline void store(unsigned *p, unsigned v) {
#if defined(GLU_SYMBOLIC_SF_HOST_TEST)
    __atomic_store_n(p, v, __ATOMIC_RELEASE);
#elif defined(GLU_TEST_CUDA_COMPAT)
    asm volatile("st.release.gpu.global.u32 [%0], %1;" : : "l"(p), "r"(v) : "memory");
#else
    __hpcc_atomic_store(p, v, __ATOMIC_RELEASE, memory_scope_device);
#endif
}
__device__ inline unsigned cas(unsigned *p, unsigned expected, unsigned value) {
#if defined(GLU_SYMBOLIC_SF_HOST_TEST)
    __atomic_compare_exchange_n(p, &expected, value, false, __ATOMIC_ACQ_REL, __ATOMIC_ACQUIRE);
#elif defined(GLU_TEST_CUDA_COMPAT)
    // Test adapter: fences surround the legacy CUDA RMW (sm_70+).
    __threadfence();
    expected = atomicCAS(p, expected, value);
    __threadfence();
#else
    __hpcc_atomic_compare_exchange_strong(p, &expected, value,
        __ATOMIC_ACQ_REL, __ATOMIC_ACQUIRE, memory_scope_device);
#endif
    return expected;
}
__device__ inline unsigned exchange(unsigned *p, unsigned value) {
#if defined(GLU_SYMBOLIC_SF_HOST_TEST)
    return __atomic_exchange_n(p, value, __ATOMIC_ACQ_REL);
#elif defined(GLU_TEST_CUDA_COMPAT)
    __threadfence();
    unsigned old = atomicExch(p, value);
    __threadfence();
    return old;
#else
    return __hpcc_atomic_exchange(p, value, __ATOMIC_ACQ_REL, memory_scope_device);
#endif
}
__device__ inline unsigned bit_or(unsigned *p, unsigned mask) {
#if defined(GLU_SYMBOLIC_SF_HOST_TEST)
    return __atomic_fetch_or(p, mask, __ATOMIC_RELAXED);
#else
    return atomicOr(p, mask);
#endif
}
__device__ inline unsigned popcount(unsigned x) {
#if defined(GLU_TEST_CUDA_COMPAT)
    return __popc(x);
#else
    return __builtin_popcount(x);
#endif
}
__device__ inline unsigned first_bit(unsigned x) { // x != 0
#if defined(GLU_TEST_CUDA_COMPAT)
    return __ffs(x) - 1;
#else
    return __builtin_ctz(x);
#endif
}

struct Stats {
    unsigned long long parks, wakeups, merges, predecessor_rows, duplicates;
    unsigned long long restores, empty_polls, closed_races, input_rows;
};
struct Context {
    unsigned n, words, workers, queue_capacity, pool_capacity, chunk_capacity;
    const unsigned *ap, *ai;
    unsigned *sym_done, *sym_state, *cursor, *wait_head, *wait_next;
    unsigned *checkpoint_head, *chunk_start, *chunk_len, *chunk_next;
    unsigned *checkpoint_pool, *final_pool, *final_start, *col_len, *diag_rel;
    unsigned *work_bitmap, *dirty_bitmap;
    unsigned *queue_value, *queue_sequence;
    // counters: queue head, tail, final used, checkpoint used, chunks used,
    // finished columns, error. No 64-bit device atomic is required.
    unsigned *counters;
    Stats *stats;
};
enum Counter { Q_HEAD, Q_TAIL, FINAL_USED, CHECKPOINT_USED, CHUNKS_USED, COMPLETED, ERROR, COUNTERS };
__device__ inline void fail(Context c, unsigned code) { cas(c.counters + ERROR, OK, code); }

// Checked CAS reservation, never an overflowing atomicAdd followed by rollback.
__device__ inline unsigned reserve(Context c, unsigned counter, unsigned len,
                                    unsigned capacity, unsigned error) {
    unsigned used = load(c.counters + counter);
    for (;;) {
        if (len > capacity - used) { fail(c, error); return EMPTY; }
        unsigned old = cas(c.counters + counter, used, used + len);
        if (old == used) return used;
        used = old;
        if (load(c.counters + ERROR)) return EMPTY;
    }
}

// Bounded MPMC ring, per-slot sequence publication and acquire consumption.
// Tickets deliberately stop before signed half-range / uint32 wrap. A run
// exceeding that limit fails explicitly rather than introducing ABA.
__device__ inline bool enqueue(Context c, unsigned column) {
    unsigned pos = load(c.counters + Q_TAIL);
    for (;;) {
        if (load(c.counters + ERROR)) return false;
        if (pos >= unsigned(INT_MAX) - c.queue_capacity) { fail(c, TICKET_LIMIT); return false; }
        unsigned slot = pos & (c.queue_capacity - 1);
        unsigned seq = load(c.queue_sequence + slot);
        int diff = static_cast<int>(seq) - static_cast<int>(pos);
        if (diff == 0) {
            unsigned old = cas(c.counters + Q_TAIL, pos, pos + 1);
            if (old == pos) {
                c.queue_value[slot] = column;
                store(c.queue_sequence + slot, pos + 1);
                return true;
            }
            pos = old;
        } else if (diff < 0) { fail(c, QUEUE_FULL); return false; }
        else pos = load(c.counters + Q_TAIL);
    }
}
__device__ inline unsigned take_column(Context c) {
    unsigned pos = load(c.counters + Q_HEAD);
    for (;;) {
        if (load(c.counters + ERROR)) return EMPTY;
        if (pos >= unsigned(INT_MAX) - c.queue_capacity) { fail(c, TICKET_LIMIT); return EMPTY; }
        unsigned slot = pos & (c.queue_capacity - 1);
        unsigned seq = load(c.queue_sequence + slot);
        int diff = static_cast<int>(seq) - static_cast<int>(pos + 1);
        if (diff == 0) {
            unsigned old = cas(c.counters + Q_HEAD, pos, pos + 1);
            if (old == pos) {
                unsigned column = c.queue_value[slot];
                store(c.queue_sequence + slot, pos + c.queue_capacity);
                if (load(c.sym_done + column) != 0 ||
                    cas(c.sym_state + column, READY, RUNNING) != READY) {
                    fail(c, BAD_STATE); return EMPTY;
                }
                return column;
            }
            pos = old;
        } else if (diff < 0) return EMPTY;
        else pos = load(c.counters + Q_HEAD);
    }
}

__device__ inline unsigned insert(unsigned *bitmap, unsigned *dirty, unsigned row) {
    unsigned mask = 1u << (row & 31), word = row >> 5;
    unsigned old = bit_or(bitmap + word, mask);
    if (!(old & mask)) bit_or(dirty + word, mask);
    return (old & mask) != 0;
}
__device__ inline unsigned find_pred(const unsigned *bitmap, unsigned cursor, unsigned col) {
    if (cursor >= col) return EMPTY;
    unsigned word = cursor >> 5;
    unsigned bits = bitmap[word] & (UINT_MAX << (cursor & 31));
    for (;;) {
        if (bits) {
            unsigned row = word * 32 + first_bit(bits);
            return row < col ? row : EMPTY;
        }
        if (++word > (col - 1) / 32) return EMPTY;
        bits = bitmap[word];
    }
}

// All helpers below are collective. Scalar control is exclusively thread 0.
struct Scratch {
    unsigned column, pred, parked, count, start, chunk, written;
    unsigned scan[256];
    unsigned long long duplicates[256];
};
__device__ inline unsigned count_rows(Context c, const unsigned *bitmap, Scratch &s,
                                       unsigned limit = UINT_MAX) {
    if (limit > c.n) limit = c.n;
    const unsigned words = (limit + 31u) / 32u;
    unsigned count = 0;
    for (unsigned w = threadIdx.x; w < words; w += blockDim.x) {
        unsigned bits = bitmap[w];
        if ((limit & 31) && w == words - 1) bits &= (1u << (limit & 31)) - 1;
        count += popcount(bits);
    }
    s.scan[threadIdx.x] = count;
    __sync_threads();
    for (unsigned stride = blockDim.x / 2; stride; stride /= 2) {
        if (threadIdx.x < stride) s.scan[threadIdx.x] += s.scan[threadIdx.x + stride];
        __sync_threads();
    }
    return s.scan[0];
}

// Scan contiguous bitmap words, then enumerate each word in increasing order.
__device__ inline void export_rows(Context c, unsigned *bitmap, unsigned *output,
                                    unsigned start, bool clear, Scratch &s) {
    if (threadIdx.x == 0) s.written = 0;
    __sync_threads();
    for (unsigned base = 0; base < c.words; base += blockDim.x) {
        unsigned w = base + threadIdx.x;
        unsigned bits = w < c.words ? bitmap[w] : 0;
        unsigned count = popcount(bits);
        s.scan[threadIdx.x] = count;
        __sync_threads();
        for (unsigned stride = 1; stride < blockDim.x; stride *= 2) {
            unsigned add = threadIdx.x >= stride ? s.scan[threadIdx.x - stride] : 0;
            __sync_threads();
            s.scan[threadIdx.x] += add;
            __sync_threads();
        }
        unsigned offset = start + s.written + s.scan[threadIdx.x] - count;
        while (bits) {
            unsigned bit = first_bit(bits);
            output[offset++] = w * 32 + bit;
            bits &= bits - 1;
        }
        if (clear && w < c.words) bitmap[w] = 0;
        __sync_threads();
        if (threadIdx.x == 0) s.written += s.scan[blockDim.x - 1];
        __sync_threads();
    }
}
__device__ inline void load_pattern(Context c, unsigned col, unsigned *bitmap,
                                    unsigned *dirty, unsigned long long &duplicates) {
    for (unsigned w = threadIdx.x; w < c.words; w += blockDim.x) bitmap[w] = dirty[w] = 0;
    __sync_threads();
    unsigned head = c.checkpoint_head[col];
    if (head == EMPTY) {
        for (size_t p = size_t(c.ap[col]) + threadIdx.x; p < c.ap[col + 1]; p += blockDim.x)
            duplicates += insert(bitmap, dirty, c.ai[p]);
        if (threadIdx.x == 0) {
            duplicates += insert(bitmap, dirty, col);
            c.stats[blockIdx.x].input_rows += c.ap[col + 1] - c.ap[col];
        }
    } else {
        for (unsigned chunk = head; chunk != EMPTY; chunk = c.chunk_next[chunk]) {
            unsigned start = c.chunk_start[chunk], len = c.chunk_len[chunk];
            for (size_t k = threadIdx.x; k < len; k += blockDim.x) {
                unsigned row = c.checkpoint_pool[start + k];
                bit_or(bitmap + (row >> 5), 1u << (row & 31));
            }
            if (threadIdx.x == 0) c.stats[blockIdx.x].restores += len;
        }
    }
    __sync_threads();
}
__device__ inline void checkpoint(Context c, unsigned col, unsigned *dirty, Scratch &s) {
    unsigned count = count_rows(c, dirty, s);
    if (threadIdx.x == 0) {
        s.count = count;
        s.start = s.chunk = EMPTY;
        if (count) {
            s.start = reserve(c, CHECKPOINT_USED, count, c.pool_capacity, CHECKPOINT_FULL);
            if (s.start != EMPTY)
                s.chunk = reserve(c, CHUNKS_USED, 1, c.chunk_capacity, CHUNKS_FULL);
        }
    }
    __sync_threads();
    if (s.chunk == EMPTY) return;
    export_rows(c, dirty, c.checkpoint_pool, s.start, true, s);
    if (threadIdx.x == 0) {
        c.chunk_start[s.chunk] = s.start;
        c.chunk_len[s.chunk] = s.count;
        c.chunk_next[s.chunk] = c.checkpoint_head[col];
        c.checkpoint_head[col] = s.chunk;
    }
    __sync_threads();
}
__device__ inline bool park(Context c, unsigned col, unsigned pred) {
    store(c.sym_done + col, 1);
    store(c.sym_state + col, WAITING);
    unsigned head = load(c.wait_head + pred);
    for (;;) {
        if (head == CLOSED) {
            // Exchange(CLOSED) is after the predecessor's release FINISHED.
            // Acquire here also publishes its rows to this continuing CTA.
            store(c.sym_done + col, 0);
            store(c.sym_state + col, RUNNING);
            ++c.stats[blockIdx.x].closed_races;
            return false;
        }
        if (load(c.counters + ERROR)) return true;
        c.wait_next[col] = head;
        unsigned old = cas(c.wait_head + pred, head, col);
        if (old == head) {
            // Never touch col's checkpoint/cursor/wait_next after this CAS:
            // a publisher can already wake it on another worker.
            ++c.stats[blockIdx.x].parks;
            return true;
        }
        head = old;
    }
}
__device__ inline void publish_and_wake(Context c, unsigned col) {
    store(c.sym_state + col, FINISHED);
    unsigned waiter = exchange(c.wait_head + col, CLOSED);
    while (waiter != EMPTY && waiter != CLOSED) {
        unsigned next = c.wait_next[waiter]; // save before another worker reuses it
        store(c.sym_done + waiter, 0);
        store(c.sym_state + waiter, READY);
        if (!enqueue(c, waiter)) return;
        ++c.stats[blockIdx.x].wakeups;
        waiter = next;
    }
    reserve(c, COMPLETED, 1, c.n, BAD_STATE);
}

__global__ void Initialize(Context c) {
    unsigned i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < c.queue_capacity) {
        c.queue_sequence[i] = i < c.n ? i + 1 : i;
        if (i < c.n) c.queue_value[i] = i;
    }
    if (i < c.n) {
        c.sym_done[i] = 0; c.sym_state[i] = READY; c.cursor[i] = 0;
        c.wait_head[i] = c.wait_next[i] = c.checkpoint_head[i] = EMPTY;
    }
    if (i < COUNTERS) c.counters[i] = i == Q_TAIL ? c.n : 0;
}
__device__ inline void Worker(Context c, Scratch &s) {
    unsigned *bitmap = c.work_bitmap + size_t(blockIdx.x) * c.words;
    unsigned *dirty = c.dirty_bitmap + size_t(blockIdx.x) * c.words;
    unsigned long long duplicates = 0;
    if (threadIdx.x == 0) c.stats[blockIdx.x] = Stats{};
    for (;;) {
        if (threadIdx.x == 0) {
            s.column = EMPTY;
            if (load(c.counters + ERROR) || load(c.counters + COMPLETED) == c.n) s.column = CLOSED;
            else s.column = take_column(c);
        }
        __sync_threads();
        if (s.column == CLOSED) break;
        if (s.column == EMPTY) {
            // Bounded scheduler backoff only; never hold a column while
            // polling its predecessor. Even one resident CTA can make progress.
            if (threadIdx.x == 0) {
                ++c.stats[blockIdx.x].empty_polls;
                for (unsigned k = 0; k < 32; ++k) if (load(c.counters + ERROR)) break;
            }
            __sync_threads();
            continue;
        }
        const unsigned col = s.column;
        load_pattern(c, col, bitmap, dirty, duplicates);
        if (threadIdx.x == 0) s.parked = 0;
        __sync_threads();
        for (;;) {
            if (threadIdx.x == 0) {
                s.pred = load(c.counters + ERROR) ? EMPTY : find_pred(bitmap, c.cursor[col], col);
            }
            __sync_threads();
            if (s.pred == EMPTY) break;
            if (threadIdx.x == 0) s.parked = load(c.sym_state + s.pred) == FINISHED ? 2 : 0;
            __sync_threads();
            if (s.parked == 0) {
                checkpoint(c, col, dirty, s);
                if (threadIdx.x == 0) s.parked = load(c.counters + ERROR) ? 1 : park(c, col, s.pred);
                __sync_threads();
                if (s.parked == 1) break;
            }
            // The leader acquired FINISHED or CLOSED; this barrier propagates
            // publication to the CTA before any lane reads the final interval.
            unsigned pred = s.pred;
            unsigned begin = c.final_start[pred] + c.diag_rel[pred];
            unsigned end = c.final_start[pred] + c.col_len[pred];
            for (size_t p = size_t(begin) + threadIdx.x; p < end; p += blockDim.x)
                duplicates += insert(bitmap, dirty, c.final_pool[p]);
            __sync_threads();
            if (threadIdx.x == 0) {
                c.cursor[col] = pred + 1;
                ++c.stats[blockIdx.x].merges;
                c.stats[blockIdx.x].predecessor_rows += end - begin;
            }
            __sync_threads();
        }
        if (s.parked == 1) continue;
        // Every block observes errors uniformly at a barrier before returning.
        if (threadIdx.x == 0) s.start = load(c.counters + ERROR);
        __sync_threads();
        if (s.start) break;
        unsigned count = count_rows(c, bitmap, s);
        if (threadIdx.x == 0) {
            s.count = count;
            s.start = reserve(c, FINAL_USED, count, c.pool_capacity, FINAL_FULL);
        }
        __sync_threads();
        if (s.start == EMPTY) break;
        export_rows(c, bitmap, c.final_pool, s.start, false, s);
        // The diagonal offset is another collective bitmap reduction; the
        // leader never scans an entire large column by itself.
        unsigned diag = count_rows(c, bitmap, s, col);
        if (threadIdx.x == 0) {
            c.final_start[col] = s.start; c.col_len[col] = s.count; c.diag_rel[col] = diag;
        }
        __sync_threads(); // join all final row/metadata writes before release
        if (threadIdx.x == 0) publish_and_wake(c, col);
        __sync_threads();
    }
    s.duplicates[threadIdx.x] = duplicates;
    __sync_threads();
    for (unsigned stride = blockDim.x / 2; stride; stride /= 2) {
        if (threadIdx.x < stride) s.duplicates[threadIdx.x] += s.duplicates[threadIdx.x + stride];
        __sync_threads();
    }
    if (threadIdx.x == 0) c.stats[blockIdx.x].duplicates = s.duplicates[0];
}
__global__ void Factorize(Context c) {
#if defined(GLU_SYMBOLIC_SF_HOST_TEST)
    Scratch s{};
#else
    __shared__ Scratch s;
#endif
    Worker(c, s);
}
__global__ void WideCounts(Context c, unsigned long long *counts) {
    unsigned i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < c.n) counts[i] = c.col_len[i];
}
__global__ void Pack(Context c, const unsigned *ptr, unsigned *rows) {
    for (unsigned col = blockIdx.x; col < c.n; col += gridDim.x)
        for (size_t p = threadIdx.x; p < c.col_len[col]; p += blockDim.x)
            rows[ptr[col] + p] = c.final_pool[c.final_start[col] + p];
}
} // namespace glu_sym_sf
#endif
