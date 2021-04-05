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
 * Visitor for recursively collecting objects in a biconnected component that
 * is determined unreachable during cycle collection.
 *
 * @ingroup libbirch
 */
class BiconnectedCollector {
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
void libbirch::BiconnectedCollector::visit(Array<T,F>& o) {
  if (!is_value<T>::value) {
    auto iter = o.begin();
    auto last = o.end();
    for (; iter != last; ++iter) {
      visit(*iter);
    }
  }
}

template<class T>
void libbirch::BiconnectedCollector::visit(Inplace<T>& o) {
  o->accept_(*this);
}

template<class T>
void libbirch::BiconnectedCollector::visit(Shared<T>& o) {
  Any* o1 = o.load();
  o.store(nullptr);
  if (o.b) {
    if (o1->decSharedBiconnected_() == 0) {
      visit(o1);
    }
  } else {
    o1->decSharedReachable_();
    visit(o1);
  }
}
