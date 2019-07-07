/**
 * @file
 *
 * Provides the standard initial pointer type according to configuration
 * options.
 */
#pragma once

#include "libbirch/InitPtr.hpp"
#include "libbirch/LazyPtr.hpp"
#include "libbirch/EagerPtr.hpp"

namespace libbirch {
/**
 * Default initial pointer type. If `ENABLE_LAZY_DEEP_CLONE` is defined true
 * this is LazyPtr<InitPtr<T>>, otherwise EagerPtr<InitPtr<T>>.
 */
#if ENABLE_LAZY_DEEP_CLONE
template<class T>
using Init = LazyPtr<InitPtr<T>>;
#else
template<class T>
using Init = EagerPtr<InitPtr<T>>;
#endif
}
