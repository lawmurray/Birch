/**
 * @file
 */
#pragma once

#include "libbirch/type.hpp"

namespace libbirch {
/**
 * Shallow thaw objects.
 */
template<class T>
void thaw(const T& o, LazyContext* context) {
  static_assert(is_value<T>::value, "unimplemented thaw()");
}
}

#if ENABLE_LAZY_DEEP_CLONE

#include "libbirch/Shared.hpp"
#include "libbirch/Weak.hpp"
#include "libbirch/Init.hpp"
#include "libbirch/Optional.hpp"
#include "libbirch/Fiber.hpp"
#include "libbirch/Array.hpp"

namespace libbirch {
template<class T>
void thaw(const Shared<T>& o, LazyContext* context) {
  o.thaw(context);
}

template<class T>
void thaw(const Weak<T>& o, LazyContext* context) {
  o.thaw(context);
}

template<class T>
void thaw(const Init<T>& o, LazyContext* context) {
  o.thaw(context);
}

template<class T>
void thaw(const Fiber<T>& o, LazyContext* context) {
  o.thaw(context);
}

template<class T, class F>
void thaw(const Array<T,F>& o, LazyContext* context) {
  if (!is_value<T>::value) {
    auto iter = o.begin();
    auto last = iter + o.size();
    for (; iter != last; ++iter) {
      thaw(*iter, context);
    }
  }
}

template<class T>
void thaw(const Optional<T>& o, LazyContext* context) {
  if (!is_value<T>::value && o.query()) {
    thaw(o.get(), context);
  }
}

template<class T>
void thaw(const std::function<T>& o) {
  assert(false);
  /// @todo Need to thaw any objects in the closure here, which may require
  /// a custom implementation of lambda functions in a similar way to fibers,
  /// rather than using std::function
}

template<int i, class ... Args>
struct thaw_tuple_impl {
  void operator()(const std::tuple<Args...>& o, LazyContext* context) {
    thaw(std::get<i - 1>(o), context);
    thaw_tuple_impl<i - 1,Args...>()(o, context);
  }
};

template<class ... Args>
struct thaw_tuple_impl<0,Args...> {
  void operator()(const std::tuple<Args...>& o, LazyContext* context) {
    //
  }
};

template<class ... Args>
void thaw(const std::tuple<Args...>& o, LazyContext* context) {
  thaw_tuple_impl<std::tuple_size<std::tuple<Args...>>::value,Args...>()(o, context);
}

}

#endif
