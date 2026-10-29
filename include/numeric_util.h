#ifndef GLU_NUMERIC_UTIL_H
#define GLU_NUMERIC_UTIL_H

#include <cstdlib>
#include <dlfcn.h>
#include <stdexcept>
#include <hc_runtime_api.h>

// Optional callbacks supplied by the profiling launcher through LD_PRELOAD.
// Resolve explicitly: non-PIE AArch64 builds can bind weak addresses to zero.

class NumericUtilScope {
public:
    NumericUtilScope() : state_(nullptr), end_(nullptr) {}
    ~NumericUtilScope() { finish(1); }
    void start(int device) {
        if (!std::getenv("GLU_NUMERIC_UTIL_SOCKET")) return;
        auto begin = reinterpret_cast<void *(*)(const char *)>(dlsym(RTLD_DEFAULT, "glu_numeric_util_begin"));
        end_ = reinterpret_cast<void (*)(void *, int)>(dlsym(RTLD_DEFAULT, "glu_numeric_util_end"));
        if (!begin || !end_)
            throw std::runtime_error("numeric utilization profiler callbacks are unavailable");
        char bdf[64]{};
        if (hcDeviceGetPCIBusId(bdf, sizeof(bdf), device) != hcSuccess)
            throw std::runtime_error("cannot obtain numeric GPU PCI address");
        state_ = begin(bdf);
    }
    void finish(int status = 0) {
        if (state_) {
            end_(state_, status);
            state_ = nullptr;
        }
    }
    NumericUtilScope(const NumericUtilScope &) = delete;
    NumericUtilScope &operator=(const NumericUtilScope &) = delete;

private:
    void *state_;
    void (*end_)(void *, int);
};

#endif
