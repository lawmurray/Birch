/**
 * @file
 */
#pragma once

#include "libbirch/EntryExitLock.hpp"
#include "libbirch/Tuple.hpp"
#include "libbirch/Array.hpp"
#include "libbirch/Optional.hpp"
#include "libbirch/Fiber.hpp"
#include "libbirch/Lazy.hpp"
#include "libbirch/Memo.hpp"

namespace libbirch {
/**
 * Global freeze lock.
 *
 * @ingroup libbirch
 */
extern EntryExitLock freezeLock;

/**
 * Visitor for freezing objects.
 *
 * @ingroup libbirch
 *
 * This should not be used directly. Instead, use the freeze() function to
 * start a freeze operation, which uses this class internally, but is thread
 * safe.
 */
class Freezer {
public:
  /**
   * Constructor.
   *
   * @param label Label of the pointer on which the freeze was initiated.
   */
  Freezer(Label* label) :
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
    auto object = o.pull();
    auto label = o.getLabel();
    if (label != this->label && label != object->getLabel()) {
      /* this is a cross pointer with an outstanding copy; must finish it
       * now, as the new memo will not have the entries required to do so
       * later */
      object = o.get();
    }
    if (object->freeze()) {
      object->freeze_(this->label);
    }
    if (label->freeze()) {
      label->freeze_(this->label);
    }
  }

  /**
   * Label of the pointer on which the freeze was initiated.
   */
  Label* label;
};

/**
 * Freeze all objects reachable from a pointer.
 *
 * @ingroup libbirch
 *
 * @param o The pointer.
 *
 * Recursively freezes all objects reachable from the pointer. This includes
 * all values in memo associated with the label, as future dereference
 * operations may produce them.
 *
 * Thread safety is achieved with an entry-exit lock. Multiple threads may
 * be freezing objects. Once one thread starts freezing an object, another
 * thread that discovers the same object will skip it. It is then necessary
 * to wait until both threads finish before either has a guarantee that all
 * reachable objects have been frozen.
 */
template<class P>
void freeze(const Lazy<P>& o) {
  freezeLock.enter();
  auto object = o.pull();
  auto label = o.getLabel();
  if (object->freeze()) {
    object->freeze_(label);
  }
  if (label->freeze()) {
    label->freeze_(label);
  }
  freezeLock.exit();
}

}
