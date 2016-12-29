/**
 * @file
 */
#pragma once

#include <cstdint>
#include <limits>

namespace bi {
/**
 * Type for sizes and indexing arithmetic. While size_t would be more usual,
 * int64_6 (or int32_t) allows direct translation from the standard types in
 * code.
 */
typedef int64_t int_t;

/**
 * Constant to indicate a mutable value. Zero is convenient here, as it
 * enables multiplication to convolve multiple values.
 *
 * @ingroup library
 */
static constexpr int_t mutable_value = 0;

/**
 * Constant to indicate default value.
 *
 * @ingroup library
 */
static constexpr int_t default_value = std::numeric_limits<int_t>::max();
}
