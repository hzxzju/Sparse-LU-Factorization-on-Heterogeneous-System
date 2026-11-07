// Optional callbacks for numeric LU and the standalone GPU spin experiment.
#include <sys/socket.h>
#include <sys/un.h>
#include <unistd.h>
#include <time.h>
#include <cerrno>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <atomic>
#include <dlfcn.h>
#include <string>

namespace {
[[noreturn]] void Fail(const char *message)
{
    std::fprintf(stderr, "[numeric-util] %s: %s\n", message, std::strerror(errno));
    std::_Exit(125);
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

struct Phase {
    int fd;
    std::string marker;
    long long begin_before, begin_after;
    long long start_ns;
    long long start_mono;
};

void Mark(const std::string &name, long long &before, long long &after)
{
    // HCTX and GPU activities share hcTracer's timeline. Bracket the marker
    // with CLOCK_REALTIME, the clock printed by ht-smi, without GPU barriers.
    using MarkFn = void (*)(const char *);
    static MarkFn mark = []() -> MarkFn {
        void *symbol = dlsym(RTLD_DEFAULT, "hctxMarkA");
        if (!symbol) {
            void *library = dlopen("libhcToolsExt.so", RTLD_NOW | RTLD_GLOBAL);
            if (library) symbol = dlsym(library, "hctxMarkA");
        }
        if (!symbol) { errno = ENOSYS; Fail("hctxMarkA unavailable"); }
        return reinterpret_cast<MarkFn>(symbol);
    }();
    before = Now(CLOCK_REALTIME);
    mark(name.c_str());
    after = Now(CLOCK_REALTIME);
}
} // namespace

extern "C" void *glu_gpu_util_begin(const char *bdf, const char *name)
{
    if (!name || std::strcmp(name, "numeric_lu_compute")) {
        errno = EINVAL; Fail("unsupported GPU utilization phase");
    }
    const char *path = std::getenv("GLU_NUMERIC_UTIL_SOCKET");
    sockaddr_un address{};
    address.sun_family = AF_UNIX;
    if (!path || std::strlen(path) >= sizeof(address.sun_path)) {
        errno = EINVAL; Fail("invalid monitor socket");
    }
    std::strcpy(address.sun_path, path);
    const int fd = socket(AF_UNIX, SOCK_STREAM | SOCK_CLOEXEC, 0);
    if (fd < 0 || connect(fd, reinterpret_cast<sockaddr *>(&address), sizeof(address)))
        Fail("connect monitor");
    char request[256];
    static std::atomic<unsigned long long> sequence{0};
    const auto id = ++sequence;
    std::snprintf(request, sizeof(request), "BEGIN\t%ld\t%llu\t%s\t%s\n",
                  static_cast<long>(getpid()), id, bdf, name);
    Send(fd, request);
    Ack(fd); // The already-running sampler's physical GPU has been checked.
    Phase *phase = new Phase{fd, "GLU_NUMERIC_UTIL:" + std::to_string(getpid()) +
                                ":" + std::to_string(id), 0, 0, 0, 0};
    Mark(phase->marker + ":begin", phase->begin_before, phase->begin_after);
    phase->start_ns = Now(CLOCK_REALTIME);
    phase->start_mono = Now(CLOCK_MONOTONIC);
    return phase;
}

extern "C" void *glu_numeric_util_begin(const char *bdf)
{
    return glu_gpu_util_begin(bdf, "numeric_lu_compute");
}

extern "C" void glu_numeric_util_end(void *state, int numeric_status)
{
    const long long end_mono = Now(CLOCK_MONOTONIC);
    const long long end_ns = Now(CLOCK_REALTIME);
    Phase *phase = static_cast<Phase *>(state);
    // The solver's existing barriers have completed all numeric streams.
    // No launch interposition, event insertion or extra synchronization here.
    long long end_before, end_after;
    Mark(phase->marker + ":end", end_before, end_after);
    char request[512];
    std::snprintf(request, sizeof(request), "END\t%lld\t%lld\t%lld\t%d\t%lld\t%lld\t%lld\t%lld\n",
                  phase->start_ns, end_ns, end_mono - phase->start_mono, numeric_status,
                  phase->begin_before, phase->begin_after, end_before, end_after);
    Send(phase->fd, request);
    Ack(phase->fd);
    close(phase->fd);
    delete phase;
}
