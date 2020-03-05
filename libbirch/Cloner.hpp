/**
 * @file
 */
#pragma once

#include "libbirch/Tuple.hpp"
#include "libbirch/Array.hpp"
#include "libbirch/Optional.hpp"
#include "libbirch/Fiber.hpp"
#include "libbirch/Lazy.hpp"
#include "libbirch/Memo.hpp"

namespace libbirch {
/**
 * Visitor for relabelling objects during a clone or recycle.
 *
 * @ingroup libbirch
 *
 * This should not be used directly. It is used internally by the clone()
 * and recycle() functions to recursively visit reachable objects.
 */
class Cloner {
public:
  /**
   * Constructor.
   */
  Cloner(Label* oldLabel, Label* newLabel) :
      oldLabel(oldLabel),
      newLabel(newLabel) {
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
   * Visit an array.
   */
  template<class T, class F>
  void visit(Array<T,F>& o) const {
    o.accept_(*this);
  }

  /**
   * Visit an optional.
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
    if (o.getLabel() == oldLabel) {
      /* objects with the old label can be lazily copied */
      o.setLabel(newLabel);
    } else {
      /* objects with any other label must be eagerly copied */
      auto object = static_cast<typename P::value_type*>(o.pull()->copy_());
      visit(object);
      o = Lazy<P>(P(object), newLabel);
    }
  }

  /**
   * Visit a raw pointer.
   */
  void visit(Any* o) const {
    o->setLabel(newLabel);
    o->accept_(*this);
  }

  /**
   * Visit a memo.
   */
  void visit(Memo& o) const {
    o.accept_(*this);
  }

private:
  /**
   * Old label, being replaced.
   */
  Label* oldLabel;

  /**
   * New label.
   */
  Label* newLabel;
};

/**
 * Clone an object via a pointer.
 *
 * @ingroup libbirch
 *
 * @param o The pointer.
 */
template<class P>
Lazy<P> clone(Lazy<P>& o) {
  freeze(o);
  auto object = static_cast<typename P::value_type*>(o.pull()->copy_());
  auto oldLabel = o.getLabel();
  auto newLabel = oldLabel->copy_();
  Cloner(oldLabel, newLabel).visit(object);
  return Lazy<P>(P(object), newLabel);
}

}
