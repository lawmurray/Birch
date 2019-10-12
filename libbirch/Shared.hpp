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
#include "libbirch/type.hpp"

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

template<class T>
struct is_value<Shared<T>> {
  static const bool value = false;
};

template<class T>
struct is_pointer<Shared<T>> {
  static const bool value = true;
};

template<class T>
void freeze(Shared<T>& o) {
  o.freeze();
}

template<class T>
void thaw(Shared<T>& o, LazyLabel* label) {
  o.thaw(label);
}

template<class T>
void finish(Shared<T>& o) {
  o.finish();
}
}
