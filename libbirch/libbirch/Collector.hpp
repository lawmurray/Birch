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
 * Visitor for recursively collecting objects in unreachable reference cycles.
 * 
 * This performs the `CollectWhite()` operation of @ref Bacon2001
 * "Bacon & Rajan (2001)".
 */
class Collector {
public:
  void visit() {
    //
  }

  template<class T, std::enable_if_t<
      is_visitable<T,Collector>::value,int> = 0>
  void visit(T& o) {
    return o.accept_(*this);
  }

  template<class T, std::enable_if_t<
      !is_visitable<T,Collector>::value &&
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
      !is_visitable<T,Collector>::value &&
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

  void visit(Any* o);
};
}

#include "libbirch/Shared.hpp"
#include "libbirch/Any.hpp"
#include "libbirch/BiconnectedCollector.hpp"

template<class T>
void libbirch::Collector::visit(Shared<T>& o) {
  if (!o.a && !o.b) {
    Any* o1 = o.load();
    if (o1) {
      o.store(nullptr);
      visit(o1);
    }
  }
}
