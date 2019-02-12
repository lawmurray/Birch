/**
 * @file
 */
#pragma once

#include "libbirch/config.hpp"
#include "libbirch/LazyAny.hpp"
#include "libbirch/EagerAny.hpp"

namespace bi {
class LazyAny;
class EagerAny;

#if USE_LAZY_DEEP_CLONE
using Any = LazyAny;
#else
using Any = EagerAny;
#endif
}
