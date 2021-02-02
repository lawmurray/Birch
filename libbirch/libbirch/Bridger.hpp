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
 * Visitor implementing the second pass of bridge finding.
 *
 * @ingroup libbirch
 */
class Bridger {
public:
  static constexpr int MAX = (1 << 30);

  std::tuple<int,int,int,int> visit(const int j, const int k) {
    return std::make_tuple(MAX, 0, 0, 0);
  }

  template<class Arg>
  std::tuple<int,int,int,int> visit(const int j, const int k, Arg& arg) {
    return std::make_tuple(MAX, 0, 0, 0);
  }

  template<class Arg, class... Args>
  std::tuple<int,int,int,int> visit(const int j, const int k, Arg& arg, Args&... args) {
    int l, h, m, n, l1, h1, m1, n1;
    std::tie(l, h, m, n) = visit(j, k, arg);
    std::tie(l1, h1, m1, n1) = visit(j + m, k + n, args...);
    l = std::min(l, l1);
    h = std::max(h, h1);
    m += m1;
    n += n1;
    return std::make_tuple(l, h, m, n);
  }

  template<class... Args>
  std::tuple<int,int,int,int> visit(const int j, const int k, std::tuple<Args...>& o) {
    return std::apply([&](Args&... args) { return visit(j, k, args...); }, o);
  }

  template<class T>
  std::tuple<int,int,int,int> visit(const int j, const int k, std::optional<T>& o) {
    if (o.has_value()) {
      return visit(j, k, o.value());
    } else {
      return std::make_tuple(MAX, 0, 0, 0);
    }
  }

  template<class T, class F>
  std::tuple<int,int,int,int> visit(const int j, const int k, Array<T,F>& o);

  template<class T>
  std::tuple<int,int,int,int> visit(const int j, const int k, Shared<T>& o);

  std::tuple<int,int,int,int> visit(const int j, const int k, Any* o);
};
}

#include "libbirch/Array.hpp"
#include "libbirch/Shared.hpp"
#include "libbirch/Any.hpp"

template<class T, class F>
std::tuple<int,int,int,int> libbirch::Bridger::visit(const int j, const int k, Array<T,F>& o) {
  int l = MAX, h = 0, m = 0, n = 0, l1, h1, m1, n1;
  if (!is_value<T>::value) {
    auto iter = o.begin();
    auto last = o.end();
    for (; iter != last; ++iter) {
      std::tie(l1, h1, m1, n1) = visit(j + m, k + n, *iter);
      l = std::min(l, l1);
      h = std::max(h, h1);
      m += m1;
      n += n1;
    }
  }
  return std::make_tuple(l, h, m, n);
}

template<class T>
std::tuple<int,int,int,int> libbirch::Bridger::visit(const int j, const int k, Shared<T>& o) {
  if (!o.b) {
    Any* o1 = o.ptr.load();  ///@todo Needn't be atomic
    int l, h, m, n;
    std::tie(l, h, m, n) = visit(j, k, o1);
    if (l == j && h < j + m) {
      /* is a bridge */
      o.b = true;
      o.c = true;
      n = 0;  // base case for post-order rank in biconnected component
    }
    return std::make_tuple(l, h, m, n);
  } else {
    return std::make_tuple(MAX, 0, 0, 0);
  }
}
