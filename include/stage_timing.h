#ifndef GLU_STAGE_TIMING_H
#define GLU_STAGE_TIMING_H

#include <time.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#ifndef CLOCK_MONOTONIC
#error "GLU timing requires CLOCK_MONOTONIC"
#endif

typedef enum {
    GLU_READ_PARSE, GLU_RHS_READ, GLU_PREPROCESS, GLU_ORDERING, GLU_SYMBOLIC,
    GLU_ANALYSIS_OTHER, GLU_CONTEXT_INIT, GLU_WORKSPACE_ALLOC,
    GLU_KERNEL_COMPILE, GLU_PLAN_UPLOAD, GLU_CLEANUP,
    GLU_VALUE_H2D, GLU_NUMERIC_FACTOR, GLU_TRIANGULAR_SOLVE,
    GLU_SOLUTION_D2H, GLU_REFINEMENT, GLU_SYNC,
    GLU_RESIDUAL_CHECK, GLU_OUTPUT_WRITE, GLU_LOGGING,
    GLU_STAGE_COUNT, GLU_TIMING_IDLE = GLU_STAGE_COUNT
} GluStage;

typedef struct {
    double seconds[GLU_STAGE_COUNT];
    struct timespec boundary;
    GluStage current;
    /* Diagnostic subset of refinement: prepare the host factor/solve path. */
    double factor_d2h_s;
} GluTimings;

static inline struct timespec glu_monotonic_now(void)
{
    struct timespec now;
    if (clock_gettime(CLOCK_MONOTONIC, &now) != 0) {
        perror("clock_gettime(CLOCK_MONOTONIC)");
        abort();
    }
    return now;
}

static inline double glu_seconds_between(struct timespec start, struct timespec end)
{
    return (double)(end.tv_sec - start.tv_sec)
        + (double)(end.tv_nsec - start.tv_nsec) * 1.e-9;
}

static inline void glu_timing_init(GluTimings *timings)
{
    memset(timings, 0, sizeof(*timings));
    timings->current = GLU_TIMING_IDLE;
    timings->boundary = glu_monotonic_now();
}

/* Switching pauses the previous stage: nested stages never double count. */
static inline GluStage glu_timing_switch(GluTimings *timings, GluStage stage)
{
    if (!timings) return GLU_TIMING_IDLE;
    const struct timespec now = glu_monotonic_now();
    const GluStage previous = timings->current;
    if (previous != GLU_TIMING_IDLE)
        timings->seconds[previous] += glu_seconds_between(timings->boundary, now);
    timings->boundary = now;
    timings->current = stage;
    return previous;
}

static inline double glu_timing_sum(const GluTimings *timings, GluStage first, GluStage last)
{
    double sum = 0;
    int stage;
    for (stage = first; stage <= (int)last; ++stage) sum += timings->seconds[stage];
    return sum;
}

#ifdef __cplusplus
#include <chrono>
#include <streambuf>

namespace glu_timing {
inline GluTimings *&active()
{
    static thread_local GluTimings *timings = nullptr;
    return timings;
}

class Session {
    GluTimings *previous_;
public:
    explicit Session(GluTimings &timings) : previous_(active()) { active() = &timings; }
    ~Session() { active() = previous_; }
    Session(const Session &) = delete;
    Session &operator=(const Session &) = delete;
};

class Scope {
    GluTimings *timings_;
    GluStage previous_;
public:
    explicit Scope(GluStage stage) : timings_(active()),
        previous_(glu_timing_switch(timings_, stage)) {}
    ~Scope() { glu_timing_switch(timings_, previous_); }
    Scope(const Scope &) = delete;
    Scope &operator=(const Scope &) = delete;
};

/* Compatible with chrono helpers, with an explicit POSIX clock source. */
struct MonotonicClock {
    typedef std::chrono::nanoseconds duration;
    typedef duration::rep rep;
    typedef duration::period period;
    typedef std::chrono::time_point<MonotonicClock> time_point;
    static constexpr bool is_steady = true;
    static time_point now() {
        const timespec t = glu_monotonic_now();
        return time_point(std::chrono::seconds(t.tv_sec) + duration(t.tv_nsec));
    }
};

/* Charge diagnostic writes and flushes to postprocessing, even in a stage. */
class LogBuffer : public std::streambuf {
    std::streambuf *destination_;
protected:
    std::streamsize xsputn(const char *s, std::streamsize n) override {
        Scope scope(GLU_LOGGING);
        return destination_->sputn(s, n);
    }
    int_type overflow(int_type c) override {
        Scope scope(GLU_LOGGING);
        return traits_type::eq_int_type(c, traits_type::eof())
            ? traits_type::not_eof(c) : destination_->sputc(traits_type::to_char_type(c));
    }
    int sync() override {
        Scope scope(GLU_LOGGING);
        return destination_->pubsync();
    }
public:
    explicit LogBuffer(std::streambuf *destination) : destination_(destination) {}
};
} // namespace glu_timing
#endif
#endif
