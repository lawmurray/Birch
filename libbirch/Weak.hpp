/**
 * @file
 */
#pragma once

#include "libbirch/config.hpp"
#include "libbirch/WeakCOW.hpp"

namespace bi {
/**
 * Weak pointer type
 */
#if USE_LAZY_DEEP_CLONE
template<class T>
using Weak = WeakCOW<T>;
#else
template<class T>
using Weak = WeakPtr<T>;
#endif
}
