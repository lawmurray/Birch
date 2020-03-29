/**
 * @file
 */
#pragma once

#include "libbirch/Tuple.hpp"
#include "libbirch/Array.hpp"
#include "libbirch/Optional.hpp"
#include "libbirch/Fiber.hpp"
#include "libbirch/Lazy.hpp"
#include "libbirch/Freezer.hpp"

namespace libbirch {
/**
 * Visitor for relabelling members of a newly copied object.
 *
 * @ingroup libbirch
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

/**
 * Clone an object via a pointer.
 *
 * @ingroup libbirch
 *
 * @param o The pointer.
 */
template<class P>
auto clone(const Lazy<P>& o) {
  auto object = o.pull();
  auto label = o.getLabel();

  object->freeze(label);

  label = new Label(*label);
  object = label->forward(object);

  return Lazy<P>(object, label);
}

}
