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
 * Visitor for recursively flagging reachable objects for cycle collection.
 *
 * This performs the `ScanBlack()` operation of @ref Bacon2001
 * "Bacon & Rajan (2001)".
 */
class Reacher {
public:
  void visit() {
    //
  }

  template<class T, std::enable_if_t<
      is_visitable<T,Reacher>::value,int> = 0>
  void visit(T& o) {
    o.accept_(*this);
  }

  template<class T, std::enable_if_t<
      !is_visitable<T,Reacher>::value &&
      is_iterable<T>::value,int> = 0>
  void visit(T& o) {
    if (!std::is_trivial<typename T::value_type>::value) {
      auto iter = o.begin();
      auto last = o.end();
      for (; iter != last; ++iter) {
        visit(*iter);
      }
    }
  }

  template<class T, std::enable_if_t<
      !is_visitable<T,Reacher>::value &&
      !is_iterable<T>::value,int> = 0>
  void visit(T& o) {
    //
  }

  template<class Arg, class... Args>
  void visit(Arg& arg, Args&... args) {
    visit(arg);
    visit(args...);
  }

  template<class... Args>
  void visit(std::tuple<Args...>& o) {
    std::apply([&](Args&... args) { return visit(args...); }, o);
  }

  template<class T>
  void visit(std::optional<T>& o) {
    if (o.has_value()) {
      visit(o.value());
    }
  }

  template<class T>
  void visit(Shared<T>& o);

  template<class T>
  void visitObject(T* o);
};
}

#include "libbirch/Shared.hpp"

template<class T>
void libbirch::Reacher::visit(Shared<T>& o) {
  if (!o.a && !o.b) {
    auto o1 = o.load();
    if (o1) {
      o1->incShared_();
      visitObject(o1);
    }
  }
}

template<class T>
void libbirch::Reacher::visitObject(T* o) {
  if (!(o->f_.exchangeOr(SCANNED) & SCANNED)) {
    o->f_.maskAnd(~MARKED);  // unset for next time
  }
  if (!(o->f_.exchangeOr(REACHED) & REACHED)) {
    o->accept_(*this);
  }
}
