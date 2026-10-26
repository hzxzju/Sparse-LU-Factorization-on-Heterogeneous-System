#ifndef GLU_GPU_DEVICE_H
#define GLU_GPU_DEVICE_H

#include <hc_runtime.h>
#include <ostream>
#include <sstream>
#include <stdexcept>

namespace glu_gpu {

struct DeviceRequirements
{
    unsigned threads;
    unsigned grid_x;
    DeviceRequirements(unsigned threads = 64, unsigned grid_x = 1)
        : threads(threads), grid_x(grid_x) {}
};

// Keep the real HTHPCC calls in one adapter so failure handling can also be
// exercised with a host-only fake runtime, without requiring a broken GPU.
struct RuntimeApi
{
    hcError_t count(int *n) { return hcGetDeviceCount(n); }
    hcError_t select(int id) { return hcSetDevice(id); }
    hcError_t properties(hcDeviceProp_t *p, int id) { return hcGetDeviceProperties(p, id); }
    hcError_t allocate(void **p, size_t bytes) { return hcMalloc(p, bytes); }
    hcError_t zero(void *p, size_t bytes) { return hcMemset(p, 0, bytes); }
    hcError_t synchronize() { return hcDeviceSynchronize(); }
    hcError_t release(void *p) { return hcFree(p); }
    void clear_error() { hcGetLastError(); }
    const char *error_string(hcError_t status) { return hcGetErrorString(status); }
};

template <typename Runtime>
std::string ErrorText(Runtime &runtime, const char *operation, hcError_t status)
{
    std::ostringstream message;
    message << operation << " failed with HTHPCC status " << static_cast<int>(status)
            << " (" << runtime.error_string(status) << ')';
    return message.str();
}

template <typename Runtime>
int SelectDevice(Runtime &runtime, int preferred, const DeviceRequirements &requirements,
                 hcDeviceProp_t &properties, std::ostream &out)
{
    int count = 0;
    const hcError_t enumeration = runtime.count(&count);
    if (enumeration != hcSuccess) {
        // Physical PCI/sysfs numbers are not runtime logical device IDs. Do not
        // guess indices when the runtime cannot enumerate its visible devices.
        throw std::runtime_error(ErrorText(runtime, "hcGetDeviceCount", enumeration) +
            "; GPU enumeration failed before per-device selection. Check driver/runtime "
            "initialization and GPU device-node access (/dev/htcd, /dev/dri). "
            "Changing the device index cannot bypass this enumeration failure.");
    }
    if (count <= 0)
        throw std::runtime_error("No visible HTHPCC GPU; hcGetDeviceCount returned zero devices");

    std::ostringstream failures;
    auto try_device = [&](int id) -> bool {
        out << "GPU selection: trying device " << id << std::endl;
        auto reject = [&](const std::string &reason) -> bool {
            out << "GPU selection: skipping device " << id << ": " << reason << std::endl;
            failures << "\n  device " << id << ": " << reason;
            runtime.clear_error();
            return false;
        };
        hcError_t status = runtime.select(id);
        if (status != hcSuccess) return reject(ErrorText(runtime, "hcSetDevice", status));
        hcDeviceProp_t candidate = {};
        status = runtime.properties(&candidate, id);
        if (status != hcSuccess)
            return reject(ErrorText(runtime, "hcGetDeviceProperties", status));
        if (candidate.waveSize != 64)
            return reject("requires a native 64-thread wave");
        if (candidate.maxThreadsPerBlock <= 0 ||
            static_cast<unsigned>(candidate.maxThreadsPerBlock) < requirements.threads)
            return reject("maxThreadsPerBlock is below the required block size");
        if (candidate.maxGridSize[0] <= 0 ||
            static_cast<unsigned>(candidate.maxGridSize[0]) < requirements.grid_x)
            return reject("maxGridSize[0] is below the required grid size");

        // Selection alone need not initialize all device resources. Force a
        // small allocation and memory operation before committing this device.
        void *probe = nullptr;
        status = runtime.allocate(&probe, sizeof(unsigned));
        if (status != hcSuccess) return reject(ErrorText(runtime, "hcMalloc(probe)", status));
        if (!probe) return reject("hcMalloc(probe) returned a null pointer");
        const char *operation = "hcMemset(probe)";
        status = runtime.zero(probe, sizeof(unsigned));
        if (status == hcSuccess) {
            operation = "hcDeviceSynchronize(probe)";
            status = runtime.synchronize();
        }
        // Always release a successful allocation, including failed probes.
        const hcError_t released = runtime.release(probe);
        if (status != hcSuccess) {
            std::string reason = ErrorText(runtime, operation, status);
            if (released != hcSuccess)
                reason += "; " + ErrorText(runtime, "hcFree(probe)", released);
            return reject(reason);
        }
        if (released != hcSuccess) return reject(ErrorText(runtime, "hcFree(probe)", released));
        properties = candidate;
        out << "GPU selection: device " << id << " ready (" << candidate.name << ')'
            << std::endl;
        return true;
    };

    // Reuse the successful symbolic device for numeric/refinement work.
    if (preferred >= 0 && preferred < count && try_device(preferred)) return preferred;
    for (int id = 0; id < count; ++id) {
        if (id == preferred) continue;
        if (try_device(id)) return id;
    }
    throw std::runtime_error("No usable HTHPCC GPU after trying all " +
                             std::to_string(count) + " visible devices:" + failures.str());
}

inline int SelectDevice(int preferred, const DeviceRequirements &requirements,
                        hcDeviceProp_t &properties, std::ostream &out)
{
    RuntimeApi runtime;
    return SelectDevice(runtime, preferred, requirements, properties, out);
}

} // namespace glu_gpu
#endif
