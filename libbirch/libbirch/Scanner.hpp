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
 * Visitor for recursively scanning objects for cycle collection.
 *
 * This performs the `Scan()` operation of @ref Bacon2001
 * "Bacon & Rajan (2001)".
 */
class Scanner {
public:
  void visit() {
    //
  }

  template<class T, std::enable_if_t<
      is_visitable<T,Scanner>::value,int> = 0>
  void visit(T& o) {
    o.accept_(*this);
  }

  template<class T, std::enable_if_t<
      !is_visitable<T,Scanner>::value &&
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
      !is_visitable<T,Scanner>::value &&
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
#include "libbirch/Reacher.hpp"

template<class T>
void libbirch::Scanner::visit(Shared<T>& o) {
  if (!o.a && !o.b) {
    T* o1 = o.load();
    if (o1) {
      visitObject(o1);
    }
  }
}

template<class T>
void libbirch::Scanner::visitObject(T* o) {
  if (!(o->f_.exchangeOr(SCANNED) & SCANNED)) {
    o->f_.maskAnd(~MARKED);  // unset for next time
    if (o->numShared_() > 0) {
      if (!(o->f_.exchangeOr(REACHED) & REACHED)) {
        Reacher visitor;
        o->accept_(visitor);
      }
    } else {
      o->accept_(*this);
    }
  }
}
