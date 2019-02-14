/**
 * @file
 */
#pragma once

#include "libbirch/LazyAny.hpp"
#include "libbirch/EagerAny.hpp"

namespace bi {
class LazyAny;
class EagerAny;

#if ENABLE_LAZY_DEEP_CLONE
using Any = LazyAny;
#else
using Any = EagerAny;
#endif
}
