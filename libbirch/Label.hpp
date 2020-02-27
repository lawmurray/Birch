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
   * Update a pointer for writing.
   *
   * @param Pointer type (SharedPtr, WeakPtr or InitPtr).
   */
  template<class P>
  void get(P& o);

  /**
   * Update a pointer for reading.
   *
   * @param Pointer type (SharedPtr, WeakPtr or InitPtr).
   */
  template<class P>
  void pull(P& o);

  /**
   * Freeze all values in the memo.
   */
  void freeze();

  /**
   * Thaw the memo.
   */
  void thaw();

private:
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
   * Memo that maps source objects to clones.
   */
  Memo memo;

  /**
   * Lock.
   */
  ReaderWriterLock lock;

  /**
   * Is this frozen? Unlike regular objects, a memo can still have new entries
   * written after it is frozen, but this thaws it again.
   */
  bool frozen;
};
}

namespace bi {
  namespace type {
template<>
struct super_type<libbirch::Label> {
  using type = libbirch::Counted;
};
  }
}

inline libbirch::Label::Label() :
    frozen(false) {
  //
}

inline libbirch::Label::~Label() {
  //
}

inline libbirch::Label* libbirch::Label::fork() {
  return new Label(this);
}

template<class P>
void libbirch::Label::get(P& o) {
  if (o.query() && o->isFrozen()) {
    lock.write();
    Any* old = o.get();
    Any* ptr = get(old);
    if (ptr != old) {
      o.replace(static_cast<typename P::value_type*>(ptr));
    }
    lock.unwrite();
  }
}

template<class P>
void libbirch::Label::pull(P& o) {
  if (o.query() && o->isFrozen()) {
    lock.read();
    Any* old = o.get();
    Any* ptr = pull(old);
    lock.unread();
    if (ptr != old) {
      /* it is possible for multiple threads to try to update o
       * simultaneously, and the interleaving operations to result in
       * incorrect reference counts updates; ensure exclusive access with a
       * write lock */
      lock.write();
      o.replace(static_cast<typename P::value_type*>(ptr));
      lock.unwrite();
    }
  }
}
