// Diagnostic interposer for the hcLaunchKernel ABI used by this project.
#include <hc_runtime_api.h>
#include <cxxabi.h>
#include <dlfcn.h>
#include <sys/socket.h>
#include <sys/un.h>
#include <unistd.h>
#include <time.h>
#include <cerrno>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <string>

namespace {
[[noreturn]] void Fail(const char *message)
{
    std::fprintf(stderr, "[kernel-util] %s: %s\n", message, std::strerror(errno));
    std::_Exit(125);
}

template <typename T> T Resolve(const char *name)
{
    void *address = dlsym(RTLD_NEXT, name);
    if (!address) { errno = ENOSYS; Fail(name); }
    return reinterpret_cast<T>(address);
}

long long Now(clockid_t clock)
{
    timespec t;
    if (clock_gettime(clock, &t)) Fail("clock_gettime");
    return static_cast<long long>(t.tv_sec) * 1000000000LL + t.tv_nsec;
}

void Send(int fd, const std::string &message)
{
    size_t sent = 0;
    while (sent < message.size()) {
        const ssize_t n = send(fd, message.data() + sent, message.size() - sent, MSG_NOSIGNAL);
        if (n < 0 && errno == EINTR) continue;
        if (n <= 0) Fail("monitor disconnected");
        sent += static_cast<size_t>(n);
    }
}

void Ack(int fd)
{
    char c;
    for (;;) {
        const ssize_t n = recv(fd, &c, 1, 0);
        if (n < 0 && errno == EINTR) continue;
        if (n != 1 || c != 'O') { errno = EPROTO; Fail("monitor rejected request"); }
        return;
    }
}

std::string KernelName(const void *function)
{
    Dl_info info{};
    std::string name;
    if (dladdr(function, &info) && info.dli_sname) {
        int status = 0;
        char *demangled = abi::__cxa_demangle(info.dli_sname, nullptr, nullptr, &status);
        name = status == 0 ? demangled : info.dli_sname;
        std::free(demangled);
    } else {
        char address[40];
        std::snprintf(address, sizeof(address), "%p", function);
        name = address;
    }
    for (char &c : name) if (c == '\t' || c == '\n' || c == '\r') c = ' ';
    return name;
}
} // namespace

extern "C" hcError_t hcLaunchKernel(const void *function, dim3 grid, dim3 block,
                                     void **args, size_t shared, hcStream_t stream)
{
    static auto launch = Resolve<decltype(&hcLaunchKernel)>("hcLaunchKernel");
    // A runtime's own internal call must not start another monitor or deadlock.
    static thread_local bool inside = false;
    if (inside) return launch(function, grid, block, args, shared, stream);
    static std::mutex mutex;
    std::lock_guard<std::mutex> lock(mutex);
    inside = true;
    struct Reset { bool &flag; ~Reset() { flag = false; } } reset{inside};

    static auto synchronize = Resolve<decltype(&hcDeviceSynchronize)>("hcDeviceSynchronize");
    static auto create = Resolve<hcError_t (*)(hcEvent_t *)>("hcEventCreate");
    static auto record = Resolve<decltype(&hcEventRecord)>("hcEventRecord");
    static auto wait = Resolve<decltype(&hcEventSynchronize)>("hcEventSynchronize");
    static auto elapsed = Resolve<decltype(&hcEventElapsedTime)>("hcEventElapsedTime");
    static auto destroy = Resolve<decltype(&hcEventDestroy)>("hcEventDestroy");
    static auto get_device = Resolve<decltype(&hcGetDevice)>("hcGetDevice");
    static auto get_bdf = Resolve<decltype(&hcDeviceGetPCIBusId)>("hcDeviceGetPCIBusId");

    hcError_t status = synchronize(); // Drain earlier work in this context.
    if (status != hcSuccess) return status;
    int device = -1;
    char bdf[64]{};
    status = get_device(&device);
    if (status != hcSuccess) return status;
    status = get_bdf(bdf, sizeof(bdf), device);
    if (status != hcSuccess) return status;
    hcEvent_t begin{}, end{};
    status = create(&begin);
    if (status != hcSuccess) return status;
    status = create(&end);
    if (status != hcSuccess) { destroy(begin); return status; }

    const char *path = std::getenv("GLU_KERNEL_UTIL_SOCKET");
    sockaddr_un address{};
    address.sun_family = AF_UNIX;
    if (!path || std::strlen(path) >= sizeof(address.sun_path)) {
        errno = EINVAL; Fail("invalid monitor socket");
    }
    std::strcpy(address.sun_path, path);
    const int fd = socket(AF_UNIX, SOCK_STREAM | SOCK_CLOEXEC, 0);
    if (fd < 0 || connect(fd, reinterpret_cast<sockaddr *>(&address), sizeof(address)))
        Fail("connect monitor");
    char request[512];
    static unsigned long long sequence = 0;
    std::snprintf(request, sizeof(request), "BEGIN\t%ld\t%llu\t%ux%ux%u\t%ux%ux%u\t%zu\t%p\t",
                  static_cast<long>(getpid()), ++sequence, grid.x, grid.y, grid.z,
                  block.x, block.y, block.z, shared, static_cast<void *>(stream));
    Send(fd, std::string(request) + bdf + "\t" + KernelName(function) + "\n");
    Ack(fd); // Wait until ht-smi has produced a valid sample for the chosen GPU.

    status = record(begin, stream);
    const long long start_ns = Now(CLOCK_REALTIME);
    const long long start_mono = Now(CLOCK_MONOTONIC);
    hcError_t launch_status = status;
    hcError_t wait_status = status;
    float device_ms = -1;
    if (status == hcSuccess) {
        launch_status = launch(function, grid, block, args, shared, stream);
        if (launch_status == hcSuccess) {
            wait_status = record(end, stream);
            if (wait_status == hcSuccess) wait_status = wait(end);
            // On record failure, still join submitted work before stopping sampling.
            else synchronize();
        }
    }
    const long long end_mono = Now(CLOCK_MONOTONIC);
    const long long end_ns = Now(CLOCK_REALTIME);
    if (launch_status == hcSuccess && wait_status == hcSuccess)
        if (elapsed(&device_ms, begin, end) != hcSuccess) device_ms = -1;
    destroy(begin);
    destroy(end);
    std::snprintf(request, sizeof(request), "END\t%lld\t%lld\t%lld\t%.9g\t%d\t%d\n",
                  start_ns, end_ns, end_mono - start_mono, device_ms,
                  static_cast<int>(launch_status), static_cast<int>(wait_status));
    Send(fd, request);
    Ack(fd); // The result is flushed to summary.csv before the next launch.
    close(fd);
    return launch_status != hcSuccess ? launch_status : wait_status;
}
