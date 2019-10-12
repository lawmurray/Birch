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
#include "libbirch/type.hpp"

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

template<class T>
struct is_value<Weak<T>> {
  static const bool value = false;
};

template<class T>
struct is_pointer<Weak<T>> {
  static const bool value = true;
};

template<class T>
void freeze(Weak<T>& o) {
  o.freeze();
}

template<class T>
void thaw(Weak<T>& o, LazyLabel* label) {
  o.thaw(label);
}

template<class T>
void finish(Weak<T>& o) {
  o.finish();
}
}
