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
 * Visitor for recursively marking objects for cycle collection.
 * 
 * This performs the `MarkGray()` operation of @ref Bacon2001
 * "Bacon & Rajan (2001)".
 */
class Marker {
public:
  void visit() {
    //
  }

  template<class T, std::enable_if_t<
      is_visitable<T,Marker>::value,int> = 0>
  void visit(T& o) {
    o.accept_(*this);
  }

  template<class T, std::enable_if_t<
      !is_visitable<T,Marker>::value &&
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
      !is_visitable<T,Marker>::value &&
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
void membirch::Marker::visit(Shared<T>& o) {
  auto [ptr, bridge] = o.unpack();
  if (!bridge && ptr) {
    visitObject(ptr);
    ptr->decSharedReachable_();
  }
}
