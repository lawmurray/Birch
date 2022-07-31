/**
 * @file
 */
#pragma once

#include "libbirch/external.hpp"
#include "libbirch/internal.hpp"
#include "libbirch/type.hpp"

namespace libbirch {
/**
 * @internal
 * 
 * Visitor implementing the first pass of bridge finding.
 */
class Spanner {
public:
  std::tuple<int,int,int> visit(const int i, const int j) {
    return std::make_tuple(i, i, 0);
  }

  template<class T, std::enable_if_t<
      is_visitable<T,Spanner>::value,int> = 0>
  std::tuple<int,int,int> visit(const int i, const int j, T& o) {
    return o.accept_(*this, i, j);
  }

  template<class T, std::enable_if_t<
      !is_visitable<T,Spanner>::value &&
      is_iterable<T>::value,int> = 0>
  std::tuple<int,int,int> visit(const int i, const int j, T& o) {
    int l = i, h = i, m = 0, l1, h1, m1;
    if (!std::is_trivial<typename T::value_type>::value) {
      auto iter = o.begin();
      auto last = o.end();
      for (; iter != last; ++iter) {
        std::tie(l1, h1, m1) = visit(i, j + m, *iter);
        l = std::min(l, l1);
        h = std::max(h, h1);
        m += m1;
      }
    }
    return std::make_tuple(l, h, m);
  }

  template<class T, std::enable_if_t<
      !is_visitable<T,Spanner>::value &&
      !is_iterable<T>::value,int> = 0>
  std::tuple<int,int,int> visit(const int i, const int j, T& o) {
    return std::make_tuple(i, i, 0);
  }

  template<class Arg, class... Args>
  std::tuple<int,int,int> visit(const int i, const int j, Arg& arg, Args&... args) {
    int l, h, m, l1, h1, m1;
    std::tie(l, h, m) = visit(i, j, arg);
    std::tie(l1, h1, m1) = visit(i, j + m, args...);
    l = std::min(l, l1);
    h = std::max(h, h1);
    m += m1;
    return std::make_tuple(l, h, m);
  }

  template<class... Args>
  std::tuple<int,int,int> visit(const int i, const int j, std::tuple<Args...>& o) {
    return std::apply([&](Args&... args) { return visit(i, j, args...); }, o);
  }

  template<class T>
  std::tuple<int,int,int> visit(const int i, const int j, std::optional<T>& o) {
    if (o.has_value()) {
      return visit(i, j, o.value());
    } else {
      return std::make_tuple(i, i, 0);
    }
  }

  template<class T>
  std::tuple<int,int,int> visit(const int i, const int j, Shared<T>& o);

  std::tuple<int,int,int> visitObject(const int i, const int j, Any* o);
};
}

#include "libbirch/Shared.hpp"

template<class T>
std::tuple<int,int,int> libbirch::Spanner::visit(const int i, const int j,
    Shared<T>& o) {
  auto [ptr, bridge] = o.unpack();
  if (!bridge && ptr) {
    return visitObject(i, j, ptr);
  } else {
    return std::make_tuple(i, i, 0);
  }
}
