/**
 * @file
 */
#pragma once

#include "libbirch/external.hpp"
#include "libbirch/internal.hpp"
#include "libbirch/BiconnectedMemo.hpp"

namespace libbirch {
/**
 * @internal
 * 
 * Copy a graph of known size, such as a biconnected component.
 *
 * @ingroup libbirch
 */
class BiconnectedCopier {
public:
  /**
   * Constructor.
   * 
   * @param o The bridge head.
   */
  BiconnectedCopier(Any* o);

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

  Any* visit(Any* o);

private:
  /**
   * Memo.
   */
  BiconnectedMemo m;
};
}

#include "libbirch/Array.hpp"
#include "libbirch/Inplace.hpp"
#include "libbirch/Shared.hpp"
#include "libbirch/Any.hpp"

template<class T, class F>
void libbirch::BiconnectedCopier::visit(Array<T,F>& o) {
  if (!is_value<T>::value) {
    auto iter = o.begin();
    auto last = o.end();
    for (; iter != last; ++iter) {
      visit(*iter);
    }
  }
}

template<class T>
void libbirch::BiconnectedCopier::visit(Inplace<T>& o) {
  o->accept_(*this);
}

template<class T>
void libbirch::BiconnectedCopier::visit(Shared<T>& o) {
  if (!o.b) {
    Any* w = o.load();
    Any* u = visit(w);
    T* v = static_cast<T*>(u);
    v->incShared_();
    o.store(v);
  }
}
