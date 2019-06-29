/**
 * @file
 *
 * Provides the standard shared pointer type according to configuration
 * options.
 */
#pragma once

#include "libbirch/SharedPtr.hpp"
#include "libbirch/LazyPtr.hpp"
#include "libbirch/EagerPtr.hpp"

namespace libbirch {
/**
 * Default shared pointer type. If `ENABLE_LAZY_DEEP_CLONE` is defined true
 * this is LazyPtr<SharedPtr<T>>, otherwise EagerPtr<SharedPtr<T>>.
 */
#if ENABLE_LAZY_DEEP_CLONE
template<class T>
using Shared = LazyPtr<SharedPtr<T>>;
#else
template<class T>
using Shared = EagerPtr<SharedPtr<T>>;
#endif
}
