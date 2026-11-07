#ifndef GLU_SOLUTION_ERROR_H
#define GLU_SOLUTION_ERROR_H

#include <cmath>
#include <limits>
#include <istream>
#include <stdexcept>
#include <string>
#include <vector>

// Existing KLU solution files contain one finite double per matrix row.
std::string DefaultCPUReferencePath(const std::string &matrix_path);
std::vector<double> ReadCPUReference(std::istream &input, unsigned n);
std::vector<double> ReadCPUReference(const std::string &path, unsigned n);

struct SolutionError {
    long double absolute_l2 = 0;
    long double relative_l2 = 0;
    long double reference_l2 = 0;
};

template <class Scalar>
SolutionError CompareSolutions(const std::vector<Scalar> &gpu,
                              const std::vector<double> &cpu)
{
    if (gpu.size() != cpu.size())
        throw std::invalid_argument("GPU/CPU solution sizes differ");
    SolutionError error;
    for (size_t i = 0; i < gpu.size(); ++i) {
        if (!std::isfinite(gpu[i]) || !std::isfinite(cpu[i]))
            throw std::runtime_error("Nonfinite GPU/CPU solution");
        const long double difference = static_cast<long double>(gpu[i]) - cpu[i];
        // hypot scales its arguments to avoid squaring large/small values.
        error.absolute_l2 = std::hypot(error.absolute_l2, difference);
        error.reference_l2 = std::hypot(error.reference_l2, static_cast<long double>(cpu[i]));
    }
    error.relative_l2 = error.reference_l2 != 0
        ? error.absolute_l2 / error.reference_l2
        : (error.absolute_l2 == 0 ? 0 : std::numeric_limits<long double>::infinity());
    return error;
}

#endif
