#include "solution_error.h"
#include <fstream>
#include <unistd.h>

std::string DefaultCPUReferencePath(const std::string &matrix_path)
{
    // Resolve beside the actual executable so CLI defaults also work from a
    // batch output directory or when lu_cmd was invoked through PATH/symlinks.
    std::vector<char> executable(4096);
    ssize_t length;
    while (true) {
        length = readlink("/proc/self/exe", executable.data(), executable.size());
        if (length < 0)
            throw std::runtime_error("Cannot locate lu_cmd; specify --cpu-reference FILE");
        if (static_cast<size_t>(length) < executable.size()) break;
        executable.resize(executable.size() * 2);
    }
    const std::string path(executable.data(), static_cast<size_t>(length));
    std::string stem = matrix_path.substr(matrix_path.find_last_of('/') + 1);
    const size_t dot = stem.find_last_of('.');
    if (dot != std::string::npos) stem.resize(dot);
    return path.substr(0, path.find_last_of('/') + 1)
        + "KLU_standalone/test_result/" + stem + "_x.txt";
}

std::vector<double> ReadCPUReference(std::istream &input, unsigned n)
{
    std::vector<double> x(n);
    for (unsigned i = 0; i < n; ++i) {
        if (!(input >> x[i]) || !std::isfinite(x[i]))
            throw std::runtime_error("CPU reference must contain exactly "
                + std::to_string(n) + " finite values (invalid/missing value at row "
                + std::to_string(i + 1) + ")");
    }
    input >> std::ws;
    if (!input.eof())
        throw std::runtime_error("CPU reference contains extra values or invalid trailing data");
    return x;
}

std::vector<double> ReadCPUReference(const std::string &path, unsigned n)
{
    std::ifstream input(path);
    if (!input) throw std::runtime_error("Cannot open CPU reference file: " + path);
    try {
        return ReadCPUReference(input, n);
    } catch (const std::exception &e) {
        throw std::runtime_error(path + ": " + e.what());
    }
}
