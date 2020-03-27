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
 * Visitor for recursively finishing lazy copies that involve a cross
 * pointer.
 *
 * @ingroup libbirch
 */
class Finisher {
public:
  /**
   * Constructor.
   *
   * @param label Label of the pointer on which the freeze was initiated.
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
    Any* object;
    Label* label = o.getLabel();
    if (label != this->label) {
      assert(label == rootLabel);

      /* cross pointer */
      object = o.get();
    } else {
      /* not a cross pointer */
      object = o.pull();
    }
    object->finish(this->label);
    label->finish(this->label);
  }

  /**
   * Label of the pointer on which the freeze was initiated.
   */
  Label* label;
};
}
