#include <hc_runtime.h>
#include "sf_sync.cuh"
#include <cstdio>
#include <vector>

// Exercise the production release/acquire helpers, including writes by lanes
// other than the leader. Dynamic monotone issuance is safe even when workers
// outnumber resident CTAs or tasks. A dependent task reads another lane's data.
__global__ void publication_chain(double *values, unsigned *next, unsigned *done,
                                  unsigned tasks, unsigned seed)
{
    __shared__ unsigned k;
    while (true) {
        if (threadIdx.x == 0)
            k = glu_sf::fetch_next_column(next);
        __sync_threads();
        if (k >= tasks)
            return;
        if (k > 0)
            glu_sf::wait_column_ready(done + k - 1);
        const unsigned lane = threadIdx.x;
        values[static_cast<size_t>(k) * blockDim.x + lane] = k == 0
            ? static_cast<double>(seed + lane)
            : values[static_cast<size_t>(k - 1) * blockDim.x + (lane + 37) % blockDim.x] + 1;
        glu_sf::publish_column_done(done + k);
    }
}

int main()
{
    const unsigned tasks = 257, lanes = 128;
    double *values = nullptr;
    unsigned *next = nullptr, *done = nullptr;
    auto cleanup = [&]() { hcFree(values); hcFree(next); hcFree(done); };
    auto check = [](hcError_t status) { return status == hcSuccess; };
    if (!check(hcMalloc((void **)&values, tasks * lanes * sizeof(double))) ||
        !check(hcMalloc((void **)&next, sizeof(unsigned))) ||
        !check(hcMalloc((void **)&done, tasks * sizeof(unsigned)))) {
        cleanup();
        return 1;
    }
    std::vector<double> host(tasks * lanes);
    std::vector<unsigned> flags(tasks);
    for (unsigned workers : {1u, 2u, 8u, 128u, 1024u}) {
        for (unsigned repeat = 0; repeat < 4; ++repeat) {
            const unsigned seed = 1000 * repeat;
            if (!check(hcMemset(next, 0, sizeof(unsigned))) ||
                !check(hcMemset(done, 0, tasks * sizeof(unsigned)))) {
                cleanup(); return 1;
            }
            publication_chain<<<workers, lanes>>>(values, next, done, tasks, seed);
            if (!check(hcGetLastError()) || !check(hcDeviceSynchronize()) ||
                !check(hcMemcpy(host.data(), values, host.size() * sizeof(double), hcMemcpyDeviceToHost)) ||
                !check(hcMemcpy(flags.data(), done, tasks * sizeof(unsigned), hcMemcpyDeviceToHost))) {
                cleanup(); return 1;
            }
            unsigned issued = 0;
            if (!check(hcMemcpy(&issued, next, sizeof(unsigned), hcMemcpyDeviceToHost)) ||
                issued != tasks + workers) {
                cleanup(); return 1;
            }
            for (unsigned k = 0; k < tasks; ++k) {
                if (flags[k] != 1) { cleanup(); return 1; }
                for (unsigned lane = 0; lane < lanes; ++lane) {
                    const double expected = seed + (lane + 37 * k) % lanes + k;
                    if (host[k * lanes + lane] != expected) {
                        std::printf("FAIL publication workers=%u repeat=%u task=%u lane=%u\n",
                                    workers, repeat, k, lane);
                        cleanup(); return 1;
                    }
                }
            }
        }
        std::printf("PASS publication/reset workers=%u\n", workers);
    }
    cleanup();
    return 0;
}
