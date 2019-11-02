/**
 * @file
 */
#pragma once

#include "libbirch/Any.hpp"
#include "libbirch/Counted.hpp"
#include "libbirch/SharedPtr.hpp"
#include "libbirch/Memo.hpp"

namespace libbirch {
/**
 * Label for bookkeeping lazy deep clones.
 *
 * @ingroup libbirch
 */
class Label: public Counted {
  friend class List;
public:
  using class_type_ = Label;

  /**
   * Constructor for root node.
   */
  Label();

  /**
   * Constructor for non-root node.
   *
   * @param parent Parent.
   */
  Label(Label* parent);

  /**
   * Destructor.
   */
  virtual ~Label();

  /**
   * Fork to create a new child label.
   *
   * @return The child label.
   */
  Label* fork();

  /**
   * Map an object that may not yet have been cloned, cloning it if
   * necessary.
   */
  Any* get(Any* o);

  /**
   * Map an object that may not yet have been cloned, without cloning it.
   * This is used as an optimization for read-only access.
   */
  Any* pull(Any* o);

  /**
   * Shallow copy.
   */
  Any* copy(Any* o);

  /**
   * Freeze all values in the memo.
   */
  void freeze();

  /**
   * Thaw the memo.
   */
  void thaw();

  virtual const char* name_() const {
    return "Label";
  }

private:
  /**
   * Memo that maps source objects to clones.
   */
  Memo m;

  /**
   * Is this frozen? Unlike regular objects, a memo can still have new entries
   * written after it is frozen, but this flags it as unfrozen again.
   */
  bool frozen;
};
}

inline libbirch::Label::Label() :
    frozen(false) {
  //
}

inline libbirch::Label::Label(Label* parent) :
    frozen(parent->frozen) {
  assert(parent);
  m.copy(parent->m);
}

inline libbirch::Label::~Label() {
  //
}

inline libbirch::Label* libbirch::Label::fork() {
  return new Label(this);
}
