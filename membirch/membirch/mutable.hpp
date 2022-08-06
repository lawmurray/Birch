/**
 * @file
 */
#pragma once

#include "membirch/external.hpp"

namespace membirch {
/**
 * Constant to indicate a mutable value in Length, Offset, Stride, etc. Zero
 * is convenient as it allows values to be multiplied to arrive a non-zero
 * non-mutable value overall, or zero mutable value overall.
 */
static constexpr int64_t mutable_value = 0;
}
