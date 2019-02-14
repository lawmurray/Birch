/**
 * @file
 */
#pragma once

#include "libbirch/SharedPtr.hpp"
#include "libbirch/LazyPtr.hpp"
#include "libbirch/EagerPtr.hpp"

namespace bi {
/**
 * Shared pointer type.
 */
#if ENABLE_LAZY_DEEP_CLONE
template<class T>
using Shared = LazyPtr<SharedPtr<T>>;
#else
template<class T>
using Shared = EagerPtr<SharedPtr<T>>;
#endif
}
