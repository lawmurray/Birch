/**
 * @file
 */
#pragma once

#include "membirch/external.hpp"
#include "membirch/internal.hpp"
#include "membirch/type.hpp"

namespace membirch {
/**
 * @internal
 * 
 * Visitor for recursively collecting objects in a biconnected component that
 * is determined unreachable during cycle collection.
 */
class BiconnectedCollector {
public:
  void visit() {
    //
  }

  template<class T, std::enable_if_t<
      is_visitable<T,BiconnectedCollector>::value,int> = 0>
  void visit(T& o) {
    o.accept_(*this);
  }

  template<class T, std::enable_if_t<
      !is_visitable<T,BiconnectedCollector>::value &&
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
      !is_visitable<T,BiconnectedCollector>::value &&
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

  void visitObject(Any* o);
};
}

#include "membirch/Shared.hpp"

template<class T>
void membirch::BiconnectedCollector::visit(Shared<T>& o) {
  auto [ptr, bridge] = o.unpack();
  if (!bridge && ptr) {
    visitObject(ptr);
    o.releaseBiconnected();
  }
}
