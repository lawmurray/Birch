/**
 * @file
 */
#pragma once

#include "libbirch/Any.hpp"
#include "libbirch/Memo.hpp"

namespace libbirch {
/**
 * Label for bookkeeping lazy deep clones.
 *
 * @ingroup libbirch
 */
class Label final: public Any {
public:
  /**
   * Constructor.
   */
  Label();

  /**
   * Copy constructor.
   */
  Label(const Label& o);

  /**
   * Destructor.
   */
  virtual ~Label() {
    //
  }

  virtual void freeze_(const Freezer& v) override {
    memo.freeze_(v);
  }

  virtual Label* copy_(const Copier& v) const override {
    return new Label(*this);
  }

  virtual Label* recycle_(const Recycler& v) override {
    return this;
  }

  /**
   * Update a pointer for writing.
   *
   * @param Smart pointer (SharedPtr, WeakPtr or InitPtr).
   */
  template<class P>
  auto get(P& o)  {
    auto ptr = o.get();
    if (o.query() && o->isFrozen()) {
      lock.write();
      auto old = ptr;
      ptr = static_cast<typename P::value_type*>(mapGet(old));
      if (ptr != old) {
        o.replace(ptr);
      }
      lock.unwrite();
    }
    return ptr;
  }

  /**
   * Update a pointer for reading.
   *
   * @param Smart pointer (SharedPtr, WeakPtr or InitPtr).
   */
  template<class P>
  auto pull(P& o) {
    auto ptr = o.get();
    if (o.query() && o->isFrozen()) {
      lock.read();
      auto old = ptr;
      ptr = static_cast<typename P::value_type*>(mapPull(old));
      lock.unread();
      if (ptr != old) {
        /* it is possible for multiple threads to try to update o
         * simultaneously, and the interleaving operations to result in
         * incorrect reference count updates; ensure exclusive access with a
         * write lock */
        lock.write();
        o.replace(ptr);
        lock.unwrite();
      }
    }
    return ptr;
  }

  /**
   * Forward an object.
   *
   * @param Raw pointer.
   *
   * This is used by objects which have access to their own label, and is
   * frozen can forward themselves onto a descendant.
   */
  template<class T>
  T* forward(T* ptr) {
    assert(ptr);
    T* result = ptr;
    if (ptr->isFrozen()) {
      lock.write();
      result = static_cast<T*>(mapGet(ptr));
      lock.unwrite();
    }
    return result;
  }

private:
  /**
   * Map an object that may not yet have been cloned, cloning it if
   * necessary.
   */
  Any* mapGet(Any* o);

  /**
   * Map an object that may not yet have been cloned, without cloning it.
   * This is used as an optimization for read-only access.
   */
  Any* mapPull(Any* o);

  /**
   * Memo that maps source objects to clones.
   */
  Memo memo;

  /**
   * Lock.
   */
  ReaderWriterLock lock;
};
}
