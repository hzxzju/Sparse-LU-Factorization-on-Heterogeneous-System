#include "gpu_device.h"

#include <cstring>
#include <iostream>
#include <sstream>
#include <vector>

namespace {
void Require(bool condition, const char *message)
{
    if (!condition) throw std::runtime_error(message);
}

struct FakeDevice
{
    // 1=set, 2=properties, 3=allocate, 4=memset, 5=sync, 6=free.
    int fail_step = 0;
    unsigned wave = 64, threads = 1024, grid = 65535;
    bool null_allocation = false;
};

struct FakeRuntime
{
    std::vector<FakeDevice> devices;
    std::vector<int> selections;
    hcError_t enumeration = hcSuccess;
    int active = -1, allocated = 0, freed = 0, cleared = 0;
    unsigned buffer = 0;

    explicit FakeRuntime(unsigned count) : devices(count) {}
    hcError_t fail(int step) const
    {
        return devices[active].fail_step == step ? static_cast<hcError_t>(3) : hcSuccess;
    }
    hcError_t count(int *n)
    {
        *n = enumeration == hcSuccess ? static_cast<int>(devices.size()) : 0;
        return enumeration;
    }
    hcError_t select(int id)
    {
        active = id;
        selections.push_back(id);
        return fail(1);
    }
    hcError_t properties(hcDeviceProp_t *p, int id)
    {
        Require(id == active, "properties queried on the wrong device");
        *p = hcDeviceProp_t();
        std::strcpy(p->name, "fake GPU");
        p->waveSize = devices[id].wave;
        p->maxThreadsPerBlock = devices[id].threads;
        p->maxGridSize[0] = devices[id].grid;
        return fail(2);
    }
    hcError_t allocate(void **p, size_t bytes)
    {
        Require(bytes == sizeof(unsigned), "unexpected probe size");
        *p = nullptr;
        if (fail(3) != hcSuccess) return fail(3);
        if (!devices[active].null_allocation) { *p = &buffer; ++allocated; }
        return hcSuccess;
    }
    hcError_t zero(void *p, size_t bytes)
    {
        Require(p == &buffer && bytes == sizeof(unsigned), "invalid memset probe");
        return fail(4);
    }
    hcError_t synchronize() { return fail(5); }
    hcError_t release(void *p)
    {
        Require(p == &buffer, "invalid probe free");
        ++freed;
        return fail(6);
    }
    void clear_error() { ++cleared; }
    const char *error_string(hcError_t) { return "simulated initialization error"; }
};

std::string Failure(FakeRuntime &runtime)
{
    hcDeviceProp_t properties;
    std::ostringstream out;
    try {
        glu_gpu::SelectDevice(runtime, -1, glu_gpu::DeviceRequirements(), properties, out);
    } catch (const std::runtime_error &e) { return e.what(); }
    throw std::runtime_error("expected selection failure");
}

void RunChecks()
{
    hcDeviceProp_t properties;
    std::ostringstream out;
    FakeRuntime first(3);
    Require(glu_gpu::SelectDevice(first, -1, glu_gpu::DeviceRequirements(), properties, out) == 0,
            "first usable device was not selected");
    Require(first.selections == std::vector<int>({0}) && first.allocated == first.freed,
            "selection continued after success or leaked a probe");

    FakeRuntime failures(7);
    for (int id = 0; id < 6; ++id) failures.devices[id].fail_step = id + 1;
    out.str("");
    Require(glu_gpu::SelectDevice(failures, -1, glu_gpu::DeviceRequirements(), properties, out) == 6,
            "did not skip each failed initialization stage");
    Require(failures.selections == std::vector<int>({0, 1, 2, 3, 4, 5, 6}),
            "devices were not tried in order");
    Require(failures.allocated == failures.freed && failures.cleared == 6,
            "failed probe cleanup/error clearing was incomplete");
    for (const char *operation : {"hcSetDevice", "hcGetDeviceProperties", "hcMalloc(probe)",
         "hcMemset(probe)", "hcDeviceSynchronize(probe)", "hcFree(probe)"})
        Require(out.str().find(operation) != std::string::npos, "missing failure diagnostics");

    FakeRuntime preferred(3);
    const int symbolic_device = glu_gpu::SelectDevice(preferred, 2, glu_gpu::DeviceRequirements(),
                                                     properties, out);
    const int numeric_device = glu_gpu::SelectDevice(preferred, symbolic_device,
        glu_gpu::DeviceRequirements(), properties, out);
    Require(symbolic_device == 2 && numeric_device == 2 &&
            preferred.selections == std::vector<int>({2, 2}), "preferred device was not reused");
    preferred.selections.clear();
    preferred.devices[2].fail_step = 1;
    Require(glu_gpu::SelectDevice(preferred, 2, glu_gpu::DeviceRequirements(), properties, out) == 0 &&
            preferred.selections == std::vector<int>({2, 0}), "failed preferred device did not fall back");
    preferred.selections.clear();
    Require(glu_gpu::SelectDevice(preferred, 99, glu_gpu::DeviceRequirements(), properties, out) == 0 &&
            preferred.selections == std::vector<int>({0}), "out-of-range preferred device was tried");

    FakeRuntime limits(4);
    limits.devices[0].wave = 32;
    limits.devices[1].threads = 64;
    limits.devices[2].grid = 1;
    Require(glu_gpu::SelectDevice(limits, -1, glu_gpu::DeviceRequirements(256, 2), properties, out) == 3,
            "device requirements were ignored");

    FakeRuntime all_bad(2);
    all_bad.devices[0].fail_step = 1; all_bad.devices[1].fail_step = 3;
    const std::string all_message = Failure(all_bad);
    Require(all_message.find("device 0") != std::string::npos &&
            all_message.find("device 1") != std::string::npos, "all-failed summary lost device reasons");

    FakeRuntime unavailable(4);
    unavailable.enumeration = static_cast<hcError_t>(3);
    const std::string enumeration_message = Failure(unavailable);
    Require(unavailable.selections.empty() &&
            enumeration_message.find("hcGetDeviceCount") != std::string::npos &&
            enumeration_message.find("/dev/htcd") != std::string::npos,
            "enumeration failure guessed indices or lacked diagnostics");
    FakeRuntime empty(0);
    Require(Failure(empty).find("zero devices") != std::string::npos && empty.selections.empty(),
            "zero-device enumeration was not handled");

    FakeRuntime null_probe(2);
    null_probe.devices[0].null_allocation = true;
    Require(glu_gpu::SelectDevice(null_probe, -1, glu_gpu::DeviceRequirements(), properties, out) == 1 &&
            null_probe.allocated == null_probe.freed, "null allocation did not skip to next device");
}
} // namespace

int main()
{
    try {
        RunChecks();
        std::cout << "PASS GPU selection: per-device failures, cleanup, preference, limits, "
                     "all-failed and enumeration errors (host fake runtime)\n";
    } catch (const std::exception &e) {
        std::cerr << e.what() << '\n';
        return 1;
    }
    return 0;
}
