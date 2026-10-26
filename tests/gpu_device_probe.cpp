#include "gpu_device.h"

#include <climits>
#include <iostream>
#include <string>

namespace {
int PrintStatus(const char *operation, hcError_t status)
{
    std::cout << operation << ": status=" << static_cast<int>(status)
              << " (" << hcGetErrorString(status) << ')' << std::endl;
    return status == hcSuccess ? 0 : 1;
}
}

int main(int argc, char **argv)
{
    try {
        // Use a separate process for each invocation: the first runtime call
        // is intentional, to distinguish enumeration from device selection.
        if (argc == 2 && std::string(argv[1]) == "--count") {
            int count = 0;
            std::cout << "Calling hcGetDeviceCount (first runtime call)" << std::endl;
            const hcError_t status = hcGetDeviceCount(&count);
            const int result = PrintStatus("hcGetDeviceCount", status);
            std::cout << "Visible devices: " << count << std::endl;
            return result;
        }
        if (argc == 3 && std::string(argv[1]) == "--set") {
            const std::string value(argv[2]);
            if (value.empty() || value.find_first_not_of("0123456789") != std::string::npos)
                throw std::invalid_argument("device ID must be a nonnegative integer");
            const unsigned long long id = std::stoull(value);
            if (id > INT_MAX) throw std::invalid_argument("device ID exceeds INT_MAX");
            std::cout << "Calling hcSetDevice(" << id << ") (first runtime call)" << std::endl;
            return PrintStatus("hcSetDevice", hcSetDevice(static_cast<int>(id)));
        }
        if (argc != 1) throw std::invalid_argument("usage: gpu_device_probe [--count | --set ID]");
        hcDeviceProp_t properties;
        std::cout << "Probing usable devices (enumeration followed by per-device probes)" << std::endl;
        const int id = glu_gpu::SelectDevice(-1, glu_gpu::DeviceRequirements(), properties, std::cout);
        std::cout << "Selected device " << id << ": " << properties.name
                  << ", waveSize=" << properties.waveSize
                  << ", sharedMemPerBlock=" << properties.sharedMemPerBlock << std::endl;
    } catch (const std::exception &e) {
        std::cerr << e.what() << '\n';
        return 1;
    }
    return 0;
}
