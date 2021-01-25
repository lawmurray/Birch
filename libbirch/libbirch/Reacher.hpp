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
 * Visitor for recursively flagging reachable objects for cycle collection.
 *
 * @ingroup libbirch
 * 
 * This performs the `ScanBlack()` operation of @ref Bacon2001
 * "Bacon & Rajan (2001)".
 */
class Reacher {
public:
  void visit() {
    //
  }

  template<class Arg, std::enable_if_t<!std::is_base_of<Any,Arg>::value,int> = 0>
  void visit(Arg& arg) {
    //
  }

  template<class Arg, class... Args>
  void visit(Arg& arg, Args&... args) {
    visit(arg);
    visit(args...);
  }

  template<class... Args>
  void visit(std::tuple<Args...>& o) {
    return std::apply(visit, o);
  }

  template<class T>
  void visit(std::optional<T>& o) {
    if (o.has_value()) {
      visit(o.value());
    }
  }

  template<class T, class F>
  void visit(Array<T,F>& o) {
    auto iter = o.begin();
    auto last = o.end();
    for (; iter != last; ++iter) {
      visit(*iter);
    }
  }

  template<class T>
  void visit(Shared<T>& o);

  template<class T, std::enable_if_t<std::is_base_of<Any,T>::value,int> = 0>
  void visit(T* o);
};
}

#include "libbirch/Any.hpp"

template<class T>
void libbirch::Reacher::visit(Shared<T>& o) {
  if (!is_acyclic<T>::value) {
    T* ptr = o.ptr.load();  ///@todo Needn't be atomic
    if (ptr) {
      visit(*ptr);
    }
  }
}

template<class T, std::enable_if_t<std::is_base_of<libbirch::Any,T>::value,int>>
void libbirch::Reacher::visit(T* o) {
  o->incShared();
  if (!(o->flags.exchangeOr(SCANNED) & SCANNED)) {
    o->flags.maskAnd(~MARKED);  // unset for next time
  }
  if (!(o->flags.exchangeOr(REACHED) & REACHED)) {
    o->accept_(*this);
  }
}
