/**
 * @file
 */
#pragma once

#include "libbirch/config.hpp"

namespace bi {
class Memo;

/**
 * The memo object associated with the current clone; @c nullptr if no clone
 * is underway.
 *
 * Ideally, clone operations would pass around the memo as an argument to
 * copy functions. This, however, means that copy constructors cannot be
 * used, which is especially problematic for types defined elsewhere (e.g.
 * std::tuple, boost::optional) where it is not possible to define a custom
 * constructor taking a memo as argument.
 *
 * Instead, this global variable is used.
 */
extern Memo* cloneMemo;
#pragma omp threadprivate(cloneMemo)
}
