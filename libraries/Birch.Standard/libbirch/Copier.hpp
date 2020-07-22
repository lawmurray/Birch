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
  template<class T, std::enable_if_t<is_value<T>::value &&
      std::is_trivially_copy_constructible<T>::value,int> = 0>
  void visit(T& arg) const {
    //
  }

  /**
   * Visit a value.
   */
  template<class T, std::enable_if_t<is_value<T>::value &&
      !std::is_trivially_copy_constructible<T>::value,int> = 0>
  void visit(T& arg) const {
    /* for types that do not support trivial copy, the bitwise copy is
     * invalid; we correct for this now by first performing a proper copy,
     * then emplacing the result over the bitwise copy */
    T proper(arg);
    new (&arg) T(std::move(proper));
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
    o.bitwiseFix();
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
    o.bitwiseFix(label);
  }

private:
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
  auto ptr = o.pull();
  auto label = o.getLabel();

  finish_lock.enter();
  ptr->finish(label);
  label->finish(label);
  finish_lock.exit();

  freeze_lock.enter();
  ptr->freeze();
  label->freeze();
  freeze_lock.exit();

  /* shared counts on labels are handled by Any, not Lazy; consequently we
   * need to complete the first copy in order to create a shared pointer to
   * the new label */
  auto newLabel = new Label(*label);
  auto newPtr = newLabel->copy(ptr);
  return Lazy<P>(newPtr, newLabel);
}

}
