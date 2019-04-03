/**
 * @file
 */
#pragma once

#include "libbirch/Shared.hpp"
#include "libbirch/Weak.hpp"
#include "libbirch/Optional.hpp"
#include "libbirch/Fiber.hpp"
#include "libbirch/Array.hpp"

namespace libbirch {
/**
 * Recursively finish objects. This is used when an object is lazily cloned,
 * to ensure that that object, and all other objects reachable from it, are
 * no longer modifiable.
 */
template<class T>
void finish(const T& o) {
  static_assert(is_value<T>::value, "finish for value applied to non-value");
}

template<class T>
void finish(const Shared<T>& o) {
  o.finish();
}

template<class T>
void finish(const Weak<T>& o) {
  o.finish();
}

template<class T>
void finish(const Fiber<T>& o) {
  o.finish();
}

template<class T, class F>
void finish(const Array<T,F>& o) {
  if (!is_value<T>::value) {
    auto iter = o.begin();
    auto last = iter + o.size();
    for (; iter != last; ++iter) {
      finish(*iter);
    }
  }
}

template<class T>
void finish(const std::initializer_list<T>& o) {
  if (!is_value<T>::value) {
    for (auto x : o) {
      finish(x);
    }
  }
}

template<class T>
void finish(const Optional<T>& o) {
  if (!is_value<T>::value && o.query()) {
    finish(o.get());
  }
}

template<class T>
void finish(const std::function<T>& o) {
  assert(false);
  /// @todo Need to finish any objects in the closure here, which may require
  /// a custom implementation of lambda functions in a similar way to fibers,
  /// rather than using std::function
}

template<int i, class ... Args>
struct finish_tuple_impl {
  void operator()(const std::tuple<Args...>& o) {
    finish(std::get<i - 1>(o));
    finish_tuple_impl<i - 1,Args...>()(o);
  }
};

template<class ... Args>
struct finish_tuple_impl<0,Args...> {
  void operator()(const std::tuple<Args...>& o) {
    //
  }
};

template<class ... Args>
void finish(const std::tuple<Args...>& o) {
  finish_tuple_impl<std::tuple_size<std::tuple<Args...>>::value,Args...>()(o);
}

}
