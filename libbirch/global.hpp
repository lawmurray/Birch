/**
 * @file
 */
#pragma once

#include "libbirch/config.hpp"

namespace bi {
class Memo;

/**
 * If the thread is performing a clone operation, then the memo object
 * associated with that clone, otherwise nullptr.
 */
extern Memo* cloneMemo;
#pragma omp threadprivate(cloneMemo)

/**
 * Constant to indicate a mutable value. Zero is convenient here, as it
 * enables multiplication to convolve multiple values.
 *
 * @ingroup libbirch
 */
static constexpr int64_t mutable_value = 0;
}
