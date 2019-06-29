/**
 * @file
 *
 * Provides the standard weak pointer type according to configuration
 * options.
 */
#pragma once

#include "libbirch/WeakPtr.hpp"
#include "libbirch/LazyPtr.hpp"
#include "libbirch/EagerPtr.hpp"

namespace libbirch {
/**
 * Default weak pointer type. If `ENABLE_LAZY_DEEP_CLONE` is defined true
 * this is LazyPtr<WeakPtr<T>>, otherwise EagerPtr<WeakPtr<T>>.
 */
#if ENABLE_LAZY_DEEP_CLONE
template<class T>
using Weak = LazyPtr<WeakPtr<T>>;
#else
template<class T>
using Weak = EagerPtr<WeakPtr<T>>;
#endif
}
