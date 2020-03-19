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
 * Visitor for relabelling members of a newly copied object.
 *
 * @ingroup libbirch
 *
 * This should not be used directly. It is used internally by the Any::cpoy_()
 * and Any::recycle_() member functions.
 */
class Copier {
public:
  /**
   * Constructor.
   */
  Copier(Label* label) :
      label(label) {
    //
  }

  /**
   * Visit empty list of variables (base case).
   */
  void visit() const {
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
   * Visit a value.
   */
  template<class T>
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
    o.setLabel(label);
  }

  /**
   * Label associated with the clone.
   */
  Label* label;
};

#include "libbirch/Freezer.hpp"

/**
 * Clone an object via a pointer.
 *
 * @ingroup libbirch
 *
 * @param o The pointer.
 */
template<class P>
Lazy<P> clone(const Lazy<P>& o) {
  freeze(o);
  auto label = new Label(*o.getLabel());
  auto object = label->forward(o.pull());
  auto r = Lazy<P>(P(object), label);
  assert(r->numShared() == 1);
  assert(label->numShared() == 1);
  return r;
}

}
