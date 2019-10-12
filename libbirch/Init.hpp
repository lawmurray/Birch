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

template<class T>
struct is_value<Init<T>> {
  static const bool value = false;
};

template<class T>
struct is_pointer<Init<T>> {
  static const bool value = true;
};

template<class T>
void freeze(Init<T>& o) {
  o.freeze();
}

template<class T>
void thaw(Init<T>& o, LazyLabel* label) {
  o.thaw(label);
}

template<class T>
void finish(Init<T>& o) {
  o.finish();
}

}
