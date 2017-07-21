/**
 * @file
 */
#pragma once

#include <limits>
#include <cstdint>

namespace bi {
class Heap;

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
 * Currently running fiber. If there is no currently running fiber, then
 * @c nullptr.
 */
extern Heap* currentFiber;
}
