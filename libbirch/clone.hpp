/**
 * @file
 */
#pragma once

namespace libbirch {
/**
 * Clone.
 */
template<class T>
auto clone(const T& o) {
  return o;
}
}

#include "libbirch/Shared.hpp"
#include "libbirch/Weak.hpp"
#include "libbirch/Init.hpp"
#include "libbirch/Optional.hpp"
#include "libbirch/Fiber.hpp"
#include "libbirch/Array.hpp"

namespace libbirch {
template<class T>
auto clone(const Shared<T>& o) {
  return Shared<T>(o, 0);
}

template<class T>
auto clone(const Weak<T>& o) {
  return Weak<T>(o, 0);
}

template<class T>
auto clone(const Init<T>& o) {
  return Init<T>(o, 0);
}

template<class T>
auto clone(const Fiber<T>& o) {
  return Fiber<T>(o, 0);
}

template<class T, class F>
auto clone(const Array<T,F>& o) {
  return Array<T,F>(o, 0);
}

template<class T>
auto clone(const Optional<T>& o) {
  if (o.query()) {
    return Optional<T>(clone(o.get()));
  } else {
    return Optional<T>();
  }
}

template<class T>
auto clone(const std::function<T>& o) {
  assert(false);
  /// @todo Need to clone any objects in the closure here, which may require
  /// a custom implementation of lambda functions in a similar way to fibers,
  /// rather than using std::function
}

template<int i, int n, class ... Args>
struct clone_tuple_impl {
  auto operator()(const std::tuple<Args...>& o) {
    return std::tuple_cat(std::make_tuple(clone(std::get<i>(o))), clone_tuple_impl<i + 1,n,Args...>()(o));
  }
};

template<int n, class ... Args>
struct clone_tuple_impl<n,n,Args...> {
  auto operator()(const std::tuple<Args...>& o) {
    return std::make_tuple();
  }
};

template<class ... Args>
auto clone(const std::tuple<Args...>& o) {
  return clone_tuple_impl<0,std::tuple_size<std::tuple<Args...>>::value,Args...>()(o);
}

}
