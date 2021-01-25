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
 * Visitor implementing the first pass of bridge finding.
 *
 * @ingroup libbirch
 */
class MarkClaimToucher {
public:
  int visit(const int i, const int j) {
    return 0;
  }

  template<class Arg, std::enable_if_t<!std::is_base_of<Any,Arg>::value,int> = 0>
  int visit(const int i, const int j, Arg& arg) {
    return 0;
  }

  template<class Arg, class... Args>
  int visit(const int i, const int j, Arg& arg, Args&... args) {
    int k = 0;
    k += visit(i, j + k, arg);
    k += visit(i, j + k, args...);
    return k;
  }

  template<class... Args>
  int visit(const int i, const int j, std::tuple<Args...>& o) {
    return std::apply([&](Args... args) { return visit(i, j, args...); }, o);
  }

  template<class T>
  int visit(const int i, const int j, std::optional<T>& o) {
    int k = 0;
    if (o.has_value()) {
      k += visit(i, j + k, o.value());
    }
    return k;
  }

  template<class T, class F>
  int visit(const int i, const int j, Array<T,F>& o) {
    int k = 0;
    auto iter = o.begin();
    auto last = o.end();
    for (; iter != last; ++iter) {
      k += visit(i, j + k, *iter);
    }
    return k;
  }

  template<class T>
  int visit(const int i, const int j, Shared<T>& o);

  template<class T, std::enable_if_t<std::is_base_of<Any,T>::value,int> = 0>
  int visit(const int i, const int j, T* o);
};
}

#include "libbirch/Any.hpp"

template<class T>
int libbirch::MarkClaimToucher::visit(const int i, const int j, Shared<T>& o) {
  int k = 0;
  if (!o.b) {
    T* ptr = o.ptr.load();  ///@todo Needn't be atomic
    k = visit(i, j + k, *ptr);
  }
  return k;
}

template<class T, std::enable_if_t<std::is_base_of<libbirch::Any,T>::value,int>>
int libbirch::MarkClaimToucher::visit(const int i, const int j, T* o) {
  int k = 0;  // number of descendants claimed by the thread
  o->decSharedReachable();
  if (!(o->flags.exchangeOr(CLAIMED) & CLAIMED)) {
    o->n.max(j);
    o->claimTid = get_thread_num();
    ++k;
    k += o->accept_(*this, j, j + k);
  } else if (o->claimTid == get_thread_num()) {
    /* previously claimed by the thread */
    o->n.max(i);
  } else {
    /* previously claimed by another thread, treat as external reference */
    o->n.store(std::numeric_limits<int>::max());
  }
  return k;
}
