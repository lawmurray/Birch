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
 * Visitor for recursively scanning objects for cycle collection.
 *
 * @ingroup libbirch
 * 
 * This performs the `Scan()` operation of @ref Bacon2001
 * "Bacon & Rajan (2001)".
 */
class Scanner {
public:
  void visit() {
    //
  }

  template<class Arg>
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
    std::apply([&](Args&... args) { return visit(args...); }, o);
  }

  template<class T>
  void visit(std::optional<T>& o) {
    if (o.has_value()) {
      visit(o.value());
    }
  }

  template<class T, class F>
  void visit(Array<T,F>& o);

  template<class T>
  void visit(Inplace<T>& o);

  template<class T>
  void visit(Shared<T>& o);

  void visit(Any* o);
};
}

#include "libbirch/Array.hpp"
#include "libbirch/Inplace.hpp"
#include "libbirch/Shared.hpp"
#include "libbirch/Any.hpp"

template<class T, class F>
void libbirch::Scanner::visit(Array<T,F>& o) {
  if (!is_value<T>::value) {
    auto iter = o.begin();
    auto last = o.end();
    for (; iter != last; ++iter) {
      visit(*iter);
    }
  }
}

template<class T>
void libbirch::Scanner::visit(Inplace<T>& o) {
  return o->accept_(*this);
}

template<class T>
void libbirch::Scanner::visit(Shared<T>& o) {
  if (!o.b) {
    Any* o1 = o.load();
    if (o1 && !o1->isAcyclic_()) {
      visit(o1);
    }
  }
}
