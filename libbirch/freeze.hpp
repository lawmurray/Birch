/**
 * @file
 */
#pragma once

#include "libbirch/Shared.hpp"
#include "libbirch/Weak.hpp"
#include "libbirch/Optional.hpp"
#include "libbirch/Fiber.hpp"
#include "libbirch/Array.hpp"

namespace bi {
/**
 * Recursively freeze objects. This is used when an object is lazily cloned,
 * to ensure that that object, and all other objects reachable from it, are
 * no longer modifiable.
 */
template<class T>
void freeze(const T& o) {
  static_assert(is_value<T>::value, "freeze for value applied to non-value");
}

template<class T>
void freeze(const Shared<T>& o) {
  o.freeze();
}

template<class T>
void freeze(const Weak<T>& o) {
  o.freeze();
}

template<class T>
void freeze(const Fiber<T>& o) {
  o.freeze();
}

template<class T, class F>
void freeze(const Array<T,F>& o) {
  if (!is_value<T>::value) {
    auto iter = o.begin();
    auto last = iter + o.size();
    for (; iter != last; ++iter) {
      freeze(*iter);
    }
  }
}

template<class T>
void freeze(const std::initializer_list<T>& o) {
  if (!is_value<T>::value) {
    for (auto x : o) {
      freeze(x);
    }
  }
}

template<class T>
void freeze(const Optional<T>& o) {
  if (!is_value<T>::value && o.query()) {
    freeze(o.get());
  }
}

template<class T>
void freeze(const std::function<T>& o) {
  assert(false);
  /// @todo Need to freeze any objects in the closure here, which may require
  /// a custom implementation of lambda functions in a similar way to fibers,
  /// rather than using std::function
}

template<int i, class ... Args>
struct freeze_tuple_impl {
  void operator()(const std::tuple<Args...>& o) {
    freeze(std::get<i - 1>(o));
    freeze_tuple_impl<i - 1,Args...>()(o);
  }
};

template<class ... Args>
struct freeze_tuple_impl<0,Args...> {
  void operator()(const std::tuple<Args...>& o) {
    //
  }
};

template<class ... Args>
void freeze(const std::tuple<Args...>& o) {
  freeze_tuple_impl<std::tuple_size<std::tuple<Args...>>::value,Args...>()(o);
}

}
