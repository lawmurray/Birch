/**
 * @file
 */
#pragma once

#include <limits>
#include <cstdint>
#include <cassert>

namespace bi {
/**
 * Constant to indicate a mutable value. Zero is convenient here, as it
 * enables multiplication to convolve multiple values.
 *
 * @ingroup library
 */
static constexpr int mutable_value = 0;

/**
 * Constant to indicate default value for lead.
 *
 * @ingroup library
 */
static constexpr size_t default_value = std::numeric_limits<size_t>::max();

/**
 * Greatest common divisor of two positive integers.
 */
inline size_t gcd(const size_t a, const size_t b) {
  /* pre-condition */
  assert(a > 0);
  assert(b > 0);

  size_t a1 = a, b1 = b;
  while (a1 != b1 && b1 != 0) {
    a1 = a1 % b1;
    std::swap(a1, b1);
  }
  return a1;
}
}
