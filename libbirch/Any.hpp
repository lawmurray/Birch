/**
 * @file
 */
#pragma once

#include "libbirch/external.hpp"
#include "libbirch/memory.hpp"
#include "libbirch/class.hpp"
#include "libbirch/Cloner.hpp"
#include "libbirch/Atomic.hpp"

namespace libbirch {
class Label;

/**
 * Base class for reference counted objects.
 *
 * @attention In order to work correctly, Any must be the *first* base
 * class in any inheritance hierarchy. This is particularly important when
 * multiple inheritance is used.
 *
 * @ingroup libbirch
 */
class Any {
public:
  using class_type_ = Any;
  using this_type_ = Any;

  /**
   * Constructor.
   */
  Any() :
      sharedCount(0u),
      memoValueCount(0u),
      weakCount(1u),
      memoKeyCount(1u),
      label(0),
      frozen(false),
      finished(false),
      single(false) {
    // no need to set size or tid, handled by operator new
  }

  /**
   * Destructor.
   */
  virtual ~Any() {
    assert(sharedCount.load() == 0u);
    assert(memoValueCount.load() == 0u);
  }

  /**
   * New operator.
   */
  void* operator new(std::size_t size) {
    auto ptr = (Any*)allocate(size);
    ptr->size = (unsigned)size;
    ptr->tid = get_thread_num();
    return ptr;
  }

  /**
   * Delete operator.
   */
  void operator delete(void* ptr) {
    auto counted = (Any*)ptr;
    counted->destroy();
    counted->deallocate();
  }

  /**
   * Assignment operator.
   */
  Any& operator=(const Any&) {
    return *this;
  }

  /**
   * Get the size, in bytes, of the object.
   */
  unsigned getSize() const {
    return size;
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
   * Finalizer. This is called when the memo value count reaches zero,
   * but before destruction and deallocation of the object. Object
   * resurrection is supported: if the finalizer results in a nonzero memo
   * value count, destruction and deallocation do not proceed.
   */
  virtual void finalize() {
    //
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
      holdLabel();
    }
  }

  /**
   * Decrement the shared count.
   */
  void decShared() {
    assert(numShared() > 0u);
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
    assert(numMemoValue() > 0u);
    if (--memoValueCount == 0u) {
      finalize();

      /* to support object resurrection, check the memo value count again
       * before proceeding with destruction */
      if (numMemoValue() == 0u) {
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
   * Is the object frozen? This returns true if either a freeze is in
   * progress (i.e. another thread is in the process of freezing the object),
   * or if the freeze is complete.
   */
  bool isFrozen() const {
    return frozen;
  }

  /**
   * Is the object finished?
   */
  bool isFinished() const {
    return finished;
  }

  /**
   * If frozen, at the time of freezing, was the reference count only one?
   */
  bool isSingle() const {
    return single;
  }

  /**
   * Get the label assigned to the object.
   */
  Label* getLabel() const {
    return (Label*)label;
  }

  /**
   * Set the label assigned to the object. The shared count must be greater
   * than zero.
   */
  void setLabel(Label* label) {
    assert(numShared() > 0u);
    releaseLabel();
    this->label = (intptr_t)label;
    holdLabel();
  }

  /**
   * Deep freeze.
   */
  void freeze() {
    if (!frozen) {
      frozen = true;
      auto nshared = numShared();
      single = nshared <= 1u && numWeak() <= 1u;
      if (nshared > 0u) {
        //doFreeze_();
      }
    }
  }

  /**
   * Shallow thaw to allow reuse of the object.
   *
   * @param label The new label of the object.
   */
  void thaw(Label* label) {
    setLabel(label);
    frozen = false;
    finished = false;
    single = false;
    //doThaw_(label);
  }

  /**
   * Accept function.
   */
  template<class V>
  void accept_(V& v) {
    //
  }

private:
  /**
   * Increment the shared count of the label (if not null).
   */
  void holdLabel() {
    auto label = getLabel();
    if (label) {
      reinterpret_cast<Any*>(label)->incShared();
      // ^ Label is an incomplete type here, cast as workaround
    }
  }

  /**
   * Decrement the shared count of the label (if not null). This is used
   * when the shared count for the object reduces to zero, while the memo
   * value count may still be greater than zero, in order to break any
   * reference cycles between objects and memos with the same label.
   */
  void releaseLabel() {
    auto label = getLabel();
    if (label) {
      reinterpret_cast<Any*>(label)->decShared();
      // ^ Label is an incomplete type here, cast as workaround
    }
  }

  /**
   * Destroy, but do not deallocate, the object.
   */
  void destroy() {
    assert(sharedCount.load() == 0u);
    assert(memoValueCount.load() == 0u);
    this->~Any();
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
   * Label of the object.
   */
  intptr_t label:61;

  /**
   * Is this frozen (read-only)?
   */
  bool frozen:1;

  /**
   * Is this finished?
   */
  bool finished:1;

  /**
   * If frozen, at the time of freezing, was the reference count only one?
   */
  bool single:1;

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
