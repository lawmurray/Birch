/**
 * @file
 */
#pragma once

#include "libbirch/external.hpp"
#include "libbirch/internal.hpp"
#include "libbirch/type.hpp"

#include <numbirch.hpp>

namespace libbirch {
/**
 * @internal
 * 
 * Visitor implementing the second pass of bridge finding.
 */
class Bridger {
public:
  static constexpr int MAX = std::numeric_limits<int>::max();

  std::tuple<int,int,int,int> visit(const int j, const int k) {
    return std::make_tuple(MAX, 0, 0, 0);
  }

  template<class T, std::enable_if_t<
      is_visitable<T,Bridger>::value,int> = 0>
  std::tuple<int,int,int,int> visit(const int j, const int k, T& o) {
    return o.accept_(*this, j, k);
  }

  template<class T, std::enable_if_t<
      !is_visitable<T,Bridger>::value &&
      is_iterable<T>::value,int> = 0>
  std::tuple<int,int,int,int> visit(const int j, const int k, T& o) {
    int l = MAX, h = 0, m = 0, n = 0, l1, h1, m1, n1;
    if (!std::is_trivial<typename T::value_type>::value) {
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

  template<class T, std::enable_if_t<
      !is_visitable<T,Bridger>::value &&
      !is_iterable<T>::value,int> = 0>
  std::tuple<int,int,int,int> visit(const int j, const int k, T& o) {
    return std::make_tuple(MAX, 0, 0, 0);
  }

  template<class Arg, class... Args>
  std::tuple<int,int,int,int> visit(const int j, const int k, Arg& arg,
      Args&... args) {
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
  std::tuple<int,int,int,int> visit(const int j, const int k,
      std::tuple<Args...>& o) {
    return std::apply([&](Args&... args) { return visit(j, k, args...); }, o);
  }

  template<class T>
  std::tuple<int,int,int,int> visit(const int j, const int k,
      std::optional<T>& o) {
    if (o.has_value()) {
      return visit(j, k, o.value());
    } else {
      return std::make_tuple(MAX, 0, 0, 0);
    }
  }

  template<class T>
  std::tuple<int,int,int,int> visit(const int j, const int k, Shared<T>& o);

  template<class T>
  std::tuple<int,int,int,int> visitObject(const int j, const int k, T* o);
};
}

#include "libbirch/Shared.hpp"

template<class T>
std::tuple<int,int,int,int> libbirch::Bridger::visit(const int j, const int k,
    Shared<T>& o) {
  if (!o.b) {
    int l, h, m, n;
    T* o1 = o.load();
    std::tie(l, h, m, n) = visitObject(j, k, o1);
    if (l == j && h < j + m) {
      /* is a bridge */
      o.b = true;
      n = 0;  // base case for post-order rank in biconnected component
    }
    return std::make_tuple(l, h, m, n);
  } else {
    return std::make_tuple(MAX, 0, 0, 0);
  }
}

template<class T>
std::tuple<int,int,int,int> libbirch::Bridger::visitObject(const int j,
    const int k, T* o) {
  if (o->p_ == get_thread_num()) {
    int l, h, m, n, l1, h1, m1, n1;
    o->p_ = -1;
    if (o->a_ < o->numShared_()) {
      l = 0;
      h = MAX;
    } else {
      l = o->l_;
      h = o->h_;
    }
    std::tie(l1, h1, m1, n1) = o->accept_(*this, j + 1, k);
    l = std::min(l, l1);
    h = std::max(h, h1);
    m = m1 + 1;
    n = n1 + 1;
    
    //o->a_ = 0;  // keep this, used later in BiconnectedCollector
    o->k_ = k;
    o->n_ = n;
    o->f_.maskAnd(~(CLAIMED|POSSIBLE_ROOT));
    // ^ while we're here, object is definitely reachable, so not a root
    return std::make_tuple(l, h, m, n);
  } else {
    return std::make_tuple(MAX, 0, 0, 0);
  }
}
