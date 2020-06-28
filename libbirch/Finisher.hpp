/**
 * @file
 */
#pragma once

#include "libbirch/Tuple.hpp"
#include "libbirch/Array.hpp"
#include "libbirch/Optional.hpp"
#include "libbirch/Fiber.hpp"
#include "libbirch/Lazy.hpp"

namespace libbirch {
/**
 * Visitor for finishing deep copies through cross pointers, for all
 * reachable objects.
 *
 * @ingroup libbirch
 */
class Finisher {
public:
  /**
   * Constructor.
   *
   * @param label Label of the object being visited. Lazy pointers that do
   * not have this label are identified as cross pointers, and require
   * finishing.
   */
  Finisher(Label* label) :
      label(label) {
    //
  }

  /**
   * Visit list of variables.
   *
   * @param arg First variable.
   * @param args... Remaining variables.
   */
  template<class Arg, class... Args>
  void visit(Arg& arg, Args&... args) const {
    visit(arg);
    visit(args...);
  }

  /**
   * Visit empty list of variables (base case).
   */
  void visit() const {
    //
  }

  /**
   * Visit a value.
   */
  template<class T, std::enable_if_t<is_value<T>::value,int> = 0>
  void visit(T& arg) const {
    //
  }

  /**
   * Visit a tuple.
   */
  template<class Head, class... Tail>
  void visit(Tuple<Head,Tail...>& o) const {
    o.accept_(*this);
  }

  /**
   * Visit an array of non-value type.
   */
  template<class T, class F>
  void visit(Array<T,F>& o) const {
    o.accept_(*this);
  }

  /**
   * Visit an optional of non-value type.
   */
  template<class T>
  void visit(Optional<T>& o) const {
    o.accept_(*this);
  }

  /**
   * Visit a fiber.
   */
  template<class Yield, class Return>
  void visit(Fiber<Yield,Return>& o) const {
    o.accept_(*this);
  }

  /**
   * Visit a lazy pointer.
   */
  template<class P>
  void visit(Lazy<P>& o) const {
    Shared<typename P::value_type> ptr;
    if (o.getLabel() != label) {
      /* cross pointer, finish copies with get() */
      ptr.replace(o.get());
    } else {
      /* not a cross pointer, just pull() */
      ptr.replace(o.pull());
    }
    ptr->finish(label);
  }

  /**
   * Label of the pointer on which the freeze was initiated.
   */
  Label* label;
};
}
