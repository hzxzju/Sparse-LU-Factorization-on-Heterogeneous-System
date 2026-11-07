#include <hc_runtime.h>
#include "numeric_sm_padding.h"
#include "numeric_util.h"
#include <algorithm>
#include <chrono>
#include <cmath>
#include <csignal>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>
#include <thread>

namespace {
using Clock = std::chrono::steady_clock;
volatile std::sig_atomic_t interrupted = 0;
void Interrupt(int) { interrupted = 1; }

// Separate host/device control words to avoid sharing a cache line between
// unrelated block leaders. These are control signals, not numerical data.
struct alignas(64) Signal { unsigned value; unsigned padding[15]; };

void Check(hcError_t status, const char *operation)
{
    if (status != hcSuccess)
        throw std::runtime_error(std::string(operation) + ": " + hcGetErrorString(status));
}

unsigned HostLoad(const Signal &signal)
{
    return __atomic_load_n(&signal.value, __ATOMIC_ACQUIRE);
}

void HostStop(Signal *signals)
{
    if (signals) __atomic_store_n(&signals[0].value, 1u, __ATOMIC_RELEASE);
}

// No matrix, floating-point operations, or synthetic arithmetic workload.
// By default only each block's leader polls, matching the LU worker's waits.
__global__ void SpinAllSms(Signal *signals, bool all_threads,
                           unsigned long long watchdog_ticks)
{
    extern __shared__ volatile unsigned char reservation[];
    if (threadIdx.x == 0) {
        reservation[0] = 0;
        __hpcc_atomic_store(&signals[blockIdx.x + 1].value, 1u,
                            __ATOMIC_RELEASE, memory_scope_system);
    }
    __sync_threads();
    bool timed_out = false;
    if (all_threads || threadIdx.x == 0) {
        const unsigned long long start = clock64();
        while (__hpcc_atomic_load(&signals[0].value, __ATOMIC_ACQUIRE,
                                  memory_scope_system) == 0u) {
            // Device-clock guard is a fallback only. Normal termination uses
            // the host's steady-clock five-second deadline, not GPU DVFS.
            if (clock64() - start > watchdog_ticks) {
                timed_out = true;
                break;
            }
        }
    }
    __sync_threads();
    if (threadIdx.x == 0)
        __hpcc_atomic_store(&signals[blockIdx.x + 1].value, timed_out ? 3u : 2u,
                            __ATOMIC_RELEASE, memory_scope_system);
}

struct Options { int device = 0; double seconds = .1; bool all_threads = false; };

Options Parse(int argc, char **argv)
{
    Options options;
    for (int i = 1; i < argc; ++i) {
        const std::string arg(argv[i]);
        if (arg == "--all-threads") options.all_threads = true;
        else if ((arg == "--device" || arg == "--seconds") && i + 1 < argc) {
            const std::string value(argv[++i]);
            size_t used = 0;
            if (arg == "--device") {
                options.device = std::stoi(value, &used);
                if (options.device < 0) throw std::invalid_argument("device must be nonnegative");
            } else {
                options.seconds = std::stod(value, &used);
                if (!std::isfinite(options.seconds) || options.seconds < .01 || options.seconds > 60.)
                    throw std::invalid_argument("seconds must be between 0.01 and 60");
            }
            if (used != value.size()) throw std::invalid_argument("invalid value for " + arg);
        } else throw std::invalid_argument("unknown or incomplete option: " + arg);
    }
    return options;
}
} // namespace

int main(int argc, char **argv)
{
    if (argc == 2 && (std::string(argv[1]) == "--help" || std::string(argv[1]) == "-h")) {
        std::cout << "Usage: gpu_idle_spin [--device LOGICAL_ID] [--seconds 5] [--all-threads]\n"
                     "One cooperative resident block per reported SM; no numerical computation.\n"
                     "Default: each block's leader polls; --all-threads makes every thread poll.\n"
                     "Run under profile_kernel_util.sh to sample GPU-util over GPU execution.\n";
        return 0;
    }
    Signal *host = nullptr;
    bool launched = false, synchronized = false;
    NumericUtilScope phase;
    std::signal(SIGINT, Interrupt);
    std::signal(SIGTERM, Interrupt);
    try {
        const Options options = Parse(argc, argv);
        int count = 0;
        Check(hcGetDeviceCount(&count), "hcGetDeviceCount");
        if (options.device >= count) throw std::invalid_argument("logical device is not visible");
        Check(hcSetDevice(options.device), "hcSetDevice");
        hcDeviceProp_t properties = {};
        Check(hcGetDeviceProperties(&properties, options.device), "hcGetDeviceProperties");
        char bdf[64] = {};
        Check(hcDeviceGetPCIBusId(bdf, sizeof(bdf), options.device), "hcDeviceGetPCIBusId");
        if (properties.maxThreadsPerBlock < 64 || properties.clockRate <= 0)
            throw std::runtime_error("unsupported block size or device clock rate");
        const unsigned threads = properties.maxThreadsPerBlock >= 256 ? 256u : 64u;
        hcFuncAttributes attributes = {};
        Check(hcFuncGetAttributes(&attributes, reinterpret_cast<const void *>(SpinAllSms)),
              "hcFuncGetAttributes");
        const NumericSmPaddingPlan plan = MakeNumericSmPaddingPlan(properties.cooperativeLaunch != 0,
            properties.multiProcessorCount, properties.maxGridSize[0],
            properties.sharedMemPerMultiprocessor, properties.sharedMemPerBlock,
            attributes.sharedSizeBytes);
        int resident = 0;
        Check(hcOccupancyMaxActiveBlocksPerMultiprocessor(&resident,
                  reinterpret_cast<const void *>(SpinAllSms), threads, plan.dynamic_shared_bytes),
              "hcOccupancyMaxActiveBlocksPerMultiprocessor");
        if (resident != 1) throw std::runtime_error("cannot guarantee one resident block per SM");

        Check(hcMallocHost(reinterpret_cast<void **>(&host),
                           (size_t(plan.grid_blocks) + 1) * sizeof(Signal), hcMallocHostMapped),
              "hcMallocHost(mapped signals)");
        std::memset(host, 0, (size_t(plan.grid_blocks) + 1) * sizeof(Signal));
        Signal *device = nullptr;
        Check(hcHostGetDevicePointer(reinterpret_cast<void **>(&device), host, 0),
              "hcHostGetDevicePointer");
        // clockRate is kHz. This generous guard only prevents a lost stop
        // signal from creating an unbounded kernel; the host controls duration.
        unsigned long long watchdog_ticks = static_cast<unsigned long long>(
            (options.seconds + 10.) * static_cast<double>(properties.clockRate) * 1000.);
        bool all_threads = options.all_threads;
        void *arguments[] = {&device, &all_threads, &watchdog_ticks};
        std::cout << "GPU idle spin: device=" << options.device << " bdf=" << bdf
                  << " name=" << properties.name << " sm_count=" << properties.multiProcessorCount
                  << " grid_blocks=" << plan.grid_blocks << " threads_per_block=" << threads
                  << " resident_blocks_per_sm=" << resident
                  << " dynamic_shared_bytes=" << plan.dynamic_shared_bytes
                  << " mode=" << (all_threads ? "all-threads" : "leader")
                  << " duration_seconds=" << options.seconds << std::endl;
        phase.start(options.device, "gpu_idle_spin");
        Check(hcLaunchCooperativeKernel(reinterpret_cast<const void *>(SpinAllSms),
              dim3(plan.grid_blocks), dim3(threads), arguments,
              static_cast<unsigned>(plan.dynamic_shared_bytes), nullptr), "cooperative spin launch");
        launched = true;
        Check(hcGetLastError(), "spin launch");
        const auto ready_deadline = Clock::now() + std::chrono::seconds(5);
        for (;;) {
            unsigned started = 0;
            for (unsigned i = 1; i <= plan.grid_blocks; ++i) {
                const unsigned status = HostLoad(host[i]);
                if (status > 1) throw std::runtime_error("block exited before all SMs were ready");
                started += status == 1;
            }
            if (started == plan.grid_blocks) break;
            if (interrupted) throw std::runtime_error("interrupted while waiting for SMs");
            if (Clock::now() >= ready_deadline)
                throw std::runtime_error("timed out waiting for every SM's started signal");
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
        }
        const auto spin_start = Clock::now();
        const auto deadline = spin_start + std::chrono::duration_cast<Clock::duration>(
            std::chrono::duration<double>(options.seconds));
        std::cout << "All " << plan.grid_blocks << " blocks started; holding all SMs for "
                  << options.seconds << " seconds." << std::endl;
        while (!interrupted && Clock::now() < deadline)
            std::this_thread::sleep_until(std::min(deadline, Clock::now() + std::chrono::milliseconds(10)));
        const auto stop_time = Clock::now();
        HostStop(host);
        Check(hcDeviceSynchronize(), "spin completion");
        synchronized = true;
        unsigned finished = 0;
        for (unsigned i = 1; i <= plan.grid_blocks; ++i)
            finished += HostLoad(host[i]) == 2;
        const bool success = !interrupted && finished == plan.grid_blocks;
        phase.finish(success ? 0 : 1);
        std::cout << std::fixed << std::setprecision(6)
                  << "GPU idle spin completed: started_blocks=" << plan.grid_blocks
                  << " finished_blocks=" << finished << " hold_seconds="
                  << std::chrono::duration<double>(stop_time - spin_start).count() << std::endl;
        Check(hcFreeHost(host), "hcFreeHost");
        host = nullptr;
        return interrupted ? 130 : success ? 0 : 1;
    } catch (const std::exception &error) {
        HostStop(host);
        if (launched && !synchronized) hcDeviceSynchronize();
        phase.finish(1);
        if (host) hcFreeHost(host);
        std::cerr << "GPU idle spin error: " << error.what() << std::endl;
        return interrupted ? 130 : 1;
    }
}
