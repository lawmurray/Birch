/**
 * @file
 */
#pragma once

#include "libbirch/external.hpp"
#include "libbirch/internal.hpp"

namespace libbirch {
/**
 * @internal
 * 
 * Visitor implementing the first pass of bridge finding.
 *
 * @ingroup libbirch
 */
class Spanner {
public:
  std::tuple<int,int,int> visit(const int i, const int j) {
    return std::make_tuple(i, i, 0);
  }

  template<class Arg>
  std::tuple<int,int,int> visit(const int i, const int j, Arg& arg) {
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

  template<class T, class F>
  std::tuple<int,int,int> visit(const int i, const int j, Array<T,F>& o);

  template<class T>
  std::tuple<int,int,int> visit(const int i, const int j, Shared<T>& o);

  std::tuple<int,int,int> visit(const int i, const int j, Any* o);
};
}

#include "libbirch/Array.hpp"
#include "libbirch/Shared.hpp"
#include "libbirch/Any.hpp"

template<class T, class F>
std::tuple<int,int,int> libbirch::Spanner::visit(const int i, const int j,
    Array<T,F>& o) {
  int l = i, h = i, m = 0, l1, h1, m1;
  if (!is_value<T>::value) {
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

template<class T>
std::tuple<int,int,int> libbirch::Spanner::visit(const int i, const int j,
    Shared<T>& o) {
  if (!o.b) {
    Any* o1 = o.load();
    return visit(i, j, o1);
  } else {
    return std::make_tuple(i, i, 0);
  }
}
