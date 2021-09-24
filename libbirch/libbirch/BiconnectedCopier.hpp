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

  template<class T>
  T* visitObject(T* o);

private:
  /**
   * Memo.
   */
  BiconnectedMemo m;
};
}

#include "libbirch/Shared.hpp"

template<class T>
void libbirch::BiconnectedCopier::visit(Shared<T>& o) {
  if (!o.b) {
    T* o1 = visitObject(o.load());
    o1->incShared_();
    o.store(o1);
  }
}

template<class T>
T* libbirch::BiconnectedCopier::visitObject(T* o) {
  auto& value = m.get(o);
  if (!value) {
    value = o->copy_();
    value->accept_(*this);
  }
  return static_cast<T*>(value);
}
