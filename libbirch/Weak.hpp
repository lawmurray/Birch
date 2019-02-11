/**
 * @file
 */
#pragma once

#include "libbirch/config.hpp"
#include "libbirch/WeakPtr.hpp"
#include "libbirch/LazyPtr.hpp"

namespace bi {
/**
 * Weak pointer type
 */
#if USE_LAZY_DEEP_CLONE
template<class T>
using Weak = LazyPtr<WeakPtr<T>>;
#else
template<class T>
using Weak = WeakPtr<T>;
#endif
}
