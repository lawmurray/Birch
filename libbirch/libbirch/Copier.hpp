/**
 * @file
 */
#pragma once

#include "libbirch/external.hpp"
#include "libbirch/Array.hpp"
#include "libbirch/Shared.hpp"

namespace libbirch {
/**
 * @internal
 * 
 * Visitor copying a two-connected component (island).
 *
 * @ingroup libbirch
 */
class Copier {
public:
  /**
   * Constructor.
   * 
   * @param n Size of the memo.
   */
  template<class T>
  Copier(T* o) : m(o->rank(), nullptr) {
    m.back() = o;
  }

  void visit() {
    //
  }

  template<class Arg, std::enable_if_t<!std::is_base_of<Any,Arg>::value,int> = 0>
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
    return std::apply(visit, o);
  }

  template<class T>
  void visit(std::optional<T>& o) {
    if (o.has_value()) {
      visit(o.value());
    }
  }

  template<class T, class F>
  void visit(Array<T,F>& o) {
    auto iter = o.begin();
    auto last = o.end();
    for (; iter != last; ++iter) {
      visit(*iter);
    }
  }

  template<class T>
  void visit(Shared<T>& o);

  template<class T, std::enable_if_t<std::is_base_of<Any,T>::value,int> = 0>
  void visit(T* o);

private:
  /**
   * Memo.
   */
  std::vector<libbirch::Any*,libbirch::Allocator<libbirch::Any*>> m;
};
}

template<class T>
void libbirch::Copier::visit(Shared<T>& o) {
  if (!o.b) {
    T* ptr = o.ptr.load();  ///@todo Needn't be atomic
    int n = ptr->rank() - 1;  ///@todo Needn't be atomic
    if (m[n]) {
      ptr = static_cast<T*>(m[n]);
    } else {
      ptr = static_cast<T*>(ptr->copy());
      m[n] = ptr;
      visit(ptr);
    }
    o.replace(ptr);
  }
}

template<class T, std::enable_if_t<std::is_base_of<libbirch::Any,T>::value,int>>
void libbirch::Copier::visit(T* o) {
  o->accept_(*this);
}
