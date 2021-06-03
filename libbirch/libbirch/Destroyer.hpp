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
 * Visitor for releasing shared pointers of an object.
 *
 * @ingroup libbirch
 */
class Destroyer {
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
};
}

#include "libbirch/Array.hpp"
#include "libbirch/Inplace.hpp"
#include "libbirch/Shared.hpp"

template<class T, class F>
void libbirch::Destroyer::visit(Array<T,F>& o) {
  if (!std::is_trivially_copyable<T>::value) {
    auto iter = o.begin();
    auto last = o.end();
    for (; iter != last; ++iter) {
      visit(*iter);
    }
  }
}

template<class T>
void libbirch::Destroyer::visit(Inplace<T>& o) {
  o->accept_(*this);
}

template<class T>
void libbirch::Destroyer::visit(Shared<T>& o) {
  o.release();
}
