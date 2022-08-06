/**
 * @file
 */
#pragma once

#include "membirch/external.hpp"
#include "membirch/internal.hpp"
#include "membirch/type.hpp"
#include "membirch/BiconnectedMemo.hpp"

namespace membirch {
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
  BiconnectedCopier(Any* o) : m(o) {
    //
  }

  void visit() {
    //
  }

  template<class T, std::enable_if_t<
      is_visitable<T,BiconnectedCopier>::value,int> = 0>
  void visit(T& o) {
    o.accept_(*this);
  }

  template<class T, std::enable_if_t<
      !is_visitable<T,BiconnectedCopier>::value &&
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
      !is_visitable<T,BiconnectedCopier>::value &&
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

  Any* visitObject(Any* o);

private:
  /**
   * Memo.
   */
  BiconnectedMemo m;
};
}

#include "membirch/Shared.hpp"

template<class T>
void membirch::BiconnectedCopier::visit(Shared<T>& o) {
  auto [ptr, bridge] = o.unpack();
  if (!bridge) {
    ptr = static_cast<T*>(visitObject(ptr));
    ptr->incShared_();
    o.store(ptr);
  }
}
