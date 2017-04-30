/**
 * @file
 */
#pragma once

#include <limits>
#include <cstdint>
#include <cassert>

namespace bi {
/**
 * Type for sizes and indexing arithmetic. While size_t would be more usual,
 * int64_t (or int32_t) allows direct translation from the standard types in
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

/**
 * Greatest common divisor of two positive integers.
 */
inline int_t gcd(const int_t a, const int_t b) {
  /* pre-condition */
  assert(a > 0);
  assert(b > 0);

  int_t a1 = a, b1 = b;
  while (a1 != b1 && b1 != 0) {
    a1 = a1 % b1;
    std::swap(a1, b1);
  }
  return a1;
}
}
