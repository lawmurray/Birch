/**
 * @file
 */
#pragma once

#include "libbirch/external.hpp"
#include "libbirch/memory.hpp"
#include "libbirch/class.hpp"
#include "libbirch/Atomic.hpp"

namespace libbirch {
/**
 * Base class for reference counted objects.
 *
 * @attention In order to work correctly, Counted must be the *first* base
 * class in any inheritance hierarchy. This is particularly important when
 * multiple inheritance is used.
 *
 * @ingroup libbirch
 */
class Counted {
public:
  /**
   * Constructor.
   */
  Counted() :
      sharedCount(0u),
      memoValueCount(0u),
      weakCount(1u),
      memoKeyCount(1u) {
    // no need to set size or tid, handled by operator new
  }

  /**
   * Copy constructor.
   */
  Counted(const Counted& o) : Counted() {
    //
  }

  /**
   * Assignment operator.
   */
  Counted& operator=(const Counted&) {
    return *this;
  }

  /**
   * Destructor.
   */
  virtual ~Counted() {
    assert(sharedCount.load() == 0u);
    assert(memoValueCount.load() == 0u);
  }

  /**
   * New operator.
   */
  void* operator new(std::size_t size) {
    auto ptr = (Counted*)allocate(size);
    ptr->size = (unsigned)size;
    ptr->tid = get_thread_num();
    return ptr;
  }

  /**
   * Delete operator.
   */
  void operator delete(void* ptr) {
    auto counted = (Counted*)ptr;
    counted->destroy();
    counted->deallocate();
  }

  /**
   * Get the size, in bytes, of the object.
   */
  unsigned getSize() const {
    return size;
  }

  /**
   * Increment the shared count.
   */
  void incShared() {
    if (++sharedCount == 1) {
      /* to support object resurrection, when the shared count increases from
       * zero, increment the memo value count also; this also occurs when an
       * object is first created */
      incMemoValue();
    }
  }

  /**
   * Decrement the shared count.
   */
  void decShared() {
    assert(sharedCount.load() > 0u);
    if (--sharedCount == 0u) {
      decMemoValue();
    }
  }

  /**
   * Increment the memo value count.
   */
  void incMemoValue() {
    memoValueCount.increment();
  }

  /**
   * Decrement the memo value count.
   */
  void decMemoValue() {
    assert(memoValueCount.load() > 0u);
    if (--memoValueCount == 0u) {
      finalize();

      /* to support object resurrection, check the shared count again before
       * proceeding with destruction */
      if (sharedCount.load() == 0u) {
        destroy();
        decWeak();
      }
    }
  }

  /**
   * Memo value count.
   */
  unsigned numMemoValue() const {
    return memoValueCount.load();
  }

  /**
   * Shared count.
   */
  unsigned numShared() const {
    return sharedCount.load();
  }

  /**
   * Increment the weak count.
   */
  void incWeak() {
    weakCount.increment();
  }

  /**
   * Decrement the weak count.
   */
  void decWeak() {
    assert(weakCount.load() > 0u);
    if (--weakCount == 0u) {
      decMemoKey();
    }
  }

  /**
   * Weak count.
   */
  unsigned numWeak() const {
    return weakCount.load();
  }

  /**
   * Increment the memo key count.
   */
  void incMemoKey() {
    memoKeyCount.increment();
  }

  /**
   * Decrement the memo key count.
   */
  void decMemoKey() {
    assert(memoKeyCount.load() > 0u);
    if (--memoKeyCount == 0u) {
      deallocate();
    }
  }

  /**
   * Memo key count.
   */
  unsigned numMemoKey() const {
    return memoKeyCount.load();
  }

  /**
   * Is this object reachable? An object is reachable if it contains a weak
   * count of one or more. In this case, while the object may be contained in
   * a memo as a key, it will never be queried, and is eligible for removal.
   */
  bool isReachable() const {
    return numWeak() > 0u;
  }

  /**
   * Finalizer. This is called when the shared reference count reaches zero,
   * but before destruction and deallocation of the object. Object
   * resurrection is supported: if the finalizer results in a nonzero shared
   * reference count, destruction and deallocation do not proceed.
   */
  virtual void finalize() {
    //
  }

protected:
  /**
   * Destroy, but do not deallocate, the object.
   */
  void destroy() {
    assert(sharedCount.load() == 0u);
    assert(memoValueCount.load() == 0u);
    this->~Counted();
  }

  /**
   * Deallocate the object. It should have previously been destroyed.
   */
  void deallocate() {
    assert(sharedCount.load() == 0u);
    assert(memoValueCount.load() == 0u);
    assert(weakCount.load() == 0u);
    assert(memoKeyCount.load() == 0u);
    libbirch::deallocate(this, size, tid);
  }

  /**
   * Shared count.
   */
  Atomic<unsigned> sharedCount;

  /**
   * Memo value count. This is one plus the number of times that the object
   * is held as a value in a memo. The plus one is a self-reference that is
   * released when the shared count reaches zero.
   */
  Atomic<unsigned> memoValueCount;

  /**
   * Weak count. This is one plus the number of times that the object is held
   * by a weak pointer. The plus one is a self-reference that is released
   * when the memo value count reaches zero.
   */
  Atomic<unsigned> weakCount;

  /**
   * Memo key count. This is one plus the number of times that the object
   * is held as a key in a memo. The plus one is a self-reference that is
   * released when the weak count reaches zero.
   */
  Atomic<unsigned> memoKeyCount;

  /**
   * Size of the object. This is set immediately after construction. A value
   * of zero is also indicative that the object is still being constructed.
   * Consequently, if the shared count reaches zero while the size is zero,
   * the object is not destroyed. This can happen when constructors create
   * shared pointers to `this`.
   */
  unsigned size;

  /**
   * Id of the thread associated with the object. This is used to return the
   * allocation to the correct pool after use, even when returned by a
   * different thread.
   */
  int tid;
};
}
