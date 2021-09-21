/**
 * @file
 */
#pragma once

#include "libbirch/external.hpp"
#include "libbirch/internal.hpp"
#include "libbirch/type.hpp"
#include "libbirch/Memo.hpp"

namespace libbirch {
/**
 * @internal
 * 
 * Copy a graph of unknown size.
 */
class Copier {
public:
  void visit() {
    //
  }

  template<class T, std::enable_if_t<
      is_visitable<T,Copier>::value,int> = 0>
  void visit(T& o) {
    return o.accept_(*this);
  }

  template<class T, std::enable_if_t<
      !is_visitable<T,Copier>::value &&
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
      !is_visitable<T,Copier>::value && 
      !is_iterable<T>::value,int> = 0>
  void visit(T& arg) {
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

  Any* visit(Any* o);

private:
  /**
   * Memo.
   */
  Memo m;
};
}

#include "libbirch/Shared.hpp"
#include "libbirch/Any.hpp"

template<class T>
void libbirch::Copier::visit(Shared<T>& o) {
  if (!o.b) {
    Any* u = o.load();
    T* v = static_cast<T*>(visit(u));
    v->incShared_();
    o.store(v);
  }
}
