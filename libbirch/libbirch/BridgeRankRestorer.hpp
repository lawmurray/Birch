/**
 * @file
 */
#pragma once

#include "libbirch/external.hpp"
#include "libbirch/Array.hpp"
#include "libbirch/Shared.hpp"

namespace libbirch {
/**
 * @internal
 * 
 * Visitor implementing the second pass of bridge finding.
 *
 * @ingroup libbirch
 */
class BridgeRankRestorer {
public:
  std::pair<int,int> visit(const int j) {
    return std::make_pair(0, 0);
  }

  template<class Arg, std::enable_if_t<!std::is_base_of<Any,Arg>::value,int> = 0>
  std::pair<int,int> visit(const int j, Arg& arg) {
    return std::make_pair(0, 0);
  }

  template<class Arg, class... Args>
  std::pair<int,int> visit(const int j, Arg& arg, Args&... args) {
    int k = 0, k1 = 0, h = 0, h1 = 0;
    std::tie(k1, h1) = visit(j + k, arg);
    k = k1;
    h = h1;
    std::tie(k1, h1) = visit(j + k, args...);
    k += k1;
    h = std::max(h, h1);
    return std::make_pair(k, h);
  }

  template<class... Args>
  std::pair<int,int> visit(const int j, std::tuple<Args...>& o) {
    return std::apply([&](Args&... args) { return visit(j, args...); }, o);
  }

  template<class T>
  std::pair<int,int> visit(const int j, std::optional<T>& o) {
    int k = 0, h = 0;
    if (o.has_value()) {
      std::tie(k, h) = visit(j + k, o.value());
    }
    return std::make_pair(k, h);
  }

  template<class T, class F>
  std::pair<int,int> visit(const int j, Array<T,F>& o) {
    int k = 0, k1 = 0, h = 0, h1 = 0;
    auto iter = o.begin();
    auto last = o.end();
    for (; iter != last; ++iter) {
      std::tie(k1, h1) = visit(j + k, *iter);
      k += k1;
      h = std::max(h, h1);
    }
    return std::make_pair(k, h);
  }

  template<class T>
  std::pair<int,int> visit(const int j, Shared<T>& o);

  template<class T, std::enable_if_t<std::is_base_of<Any,T>::value,int> = 0>
  std::pair<int,int> visit(const int j, T* o);
};
}

#include "libbirch/Any.hpp"

template<class T>
std::pair<int,int> libbirch::BridgeRankRestorer::visit(const int j,
    Shared<T>& o) {
  int k = 0, h = 0;
  if (!o.b) {
    T* ptr = o.ptr.load();  ///@todo Needn't be atomic
    std::tie(k, h) = visit(j + k, *ptr);
    if (h < j + k && ptr->numShared() == 0) {
      /* is a bridge */
      o.b = true;
      o.c = true;
      k = 0;  // base case for post-order rank in biconnected component
    }
  }
  return std::make_pair(k, h);
}

template<class T, std::enable_if_t<std::is_base_of<libbirch::Any,T>::value,int>>
std::pair<int,int> libbirch::BridgeRankRestorer::visit(const int j, T* o) {
  int k = 0;  // number of descendants claimed by the thread
  int h = 0;  // highest rank reachable among them
  if (o->claimTid == get_thread_num()) {
    o->flags.maskAnd(~CLAIMED);
    o->claimTid = 0;
    ++k;
    auto ret = o->accept_(*this, j + k);
    k += std::get<0>(ret);
    h = std::max(std::get<1>(ret), o->n.exchange(k));  // @todo Need not be atomic
    // ^ n now post-order rank in biconnected component
  }
  o->incShared();
  return std::make_pair(k, h);
}
