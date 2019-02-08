/**
 * @file
 */
#pragma once

#include "libbirch/config.hpp"
#include "libbirch/SharedCOW.hpp"

namespace bi {
/**
 * Shared pointer type.
 */
#if USE_LAZY_DEEP_CLONE
template<class T>
using Shared = SharedCOW<T>;
#else
template<class T>
using Shared = SharedPtr<T>;
#endif
}
