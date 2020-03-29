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

  virtual void freeze_(Label* label) override {
    lock.read();
    memo.freeze(label);
    lock.unread();
  }

  virtual Label* copy_(Label* label) const override {
    return new Label(*this);
  }

  virtual Label* recycle_(Label* label) override {
    return this;
  }

  virtual void discard_() override {
    //
  }

  virtual void restore_() override {
    //
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
      if (ptr != old) {
        o.replace(ptr);
      }
      lock.unread();
    }
    return ptr;
  }

  /**
   * Forward an object.
   *
   * @param Raw pointer.
   *
   * This is similar to get(), but used by objects which have access to their
   * own label, and if frozen can forward themselves onto a descendant.
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
