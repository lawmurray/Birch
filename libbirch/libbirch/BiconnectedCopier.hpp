/**
 * @file
 */
#pragma once

#include "libbirch/external.hpp"
#include "libbirch/internal.hpp"
#include "libbirch/type.hpp"
#include "libbirch/BiconnectedMemo.hpp"

namespace libbirch {
/**
 * @internal
 * 
 * Copy a graph of known size, such as a biconnected component.
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

  template<class Arg, std::enable_if_t<!is_iterable<Arg>::value,int> = 0>
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

  template<class T, std::enable_if_t<is_iterable<T>::value,int> = 0>
  void visit(T& o) {
    if (!std::is_trivial<typename T::value_type>::value) {
      auto iter = o.begin();
      auto last = o.end();
      for (; iter != last; ++iter) {
        visit(*iter);
      }
    }
  }

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

#include "libbirch/Inplace.hpp"
#include "libbirch/Shared.hpp"
#include "libbirch/Any.hpp"

template<class T>
void libbirch::BiconnectedCopier::visit(Inplace<T>& o) {
  o->accept_(*this);
}

template<class T>
void libbirch::BiconnectedCopier::visit(Shared<T>& o) {
  if (!o.b) {
    Any* u = o.load();
    T* v = static_cast<T*>(visit(u));
    v->incShared_();
    o.store(v);
  }
}
