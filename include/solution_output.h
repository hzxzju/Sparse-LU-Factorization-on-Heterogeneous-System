#ifndef GLU_SOLUTION_OUTPUT_H
#define GLU_SOLUTION_OUTPUT_H

#include <cmath>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

// Encode explicitly so the on-disk format is independent of host endianness.
inline void WriteSolutionBinary(const char *path, const std::vector<long double> &x)
{
    static_assert(sizeof(double) == 8 && std::numeric_limits<double>::is_iec559,
                  "x.bin requires IEEE-754 binary64");
    if (x.size() > static_cast<uint64_t>(std::numeric_limits<int64_t>::max()))
        throw std::runtime_error("Solution length exceeds int64");
    for (long double value : x)
        if (!std::isfinite(value) || !std::isfinite(static_cast<double>(value)))
            throw std::runtime_error("Solution is not finite in float64");

    std::ofstream output(path, std::ios::binary | std::ios::trunc);
    auto write_word = [&](uint64_t word) {
        char bytes[8];
        for (unsigned i = 0; i < 8; ++i)
            bytes[i] = static_cast<char>((word >> (8 * i)) & 0xff);
        output.write(bytes, sizeof(bytes));
    };
    write_word(static_cast<uint64_t>(x.size()));
    for (long double value : x) {
        const double binary64 = static_cast<double>(value);
        uint64_t bits;
        std::memcpy(&bits, &binary64, sizeof(bits));
        write_word(bits);
    }
    output.close();
    if (!output) throw std::runtime_error(std::string("Failed to write ") + path);
}

#endif
