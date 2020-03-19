/**
 * @file
 */
#pragma once

#include "libbirch/external.hpp"
#include "libbirch/memory.hpp"
#include "libbirch/Atomic.hpp"

namespace libbirch {
class Label;
class Freezer;
class Copier;
using Recycler = Copier;

/**
 * Base class for reference counted objects.
 *
 * @ingroup libbirch
 *
 * @attention In order to work correctly, Any must be the *first* base
 * class in any inheritance hierarchy. This is particularly important when
 * multiple inheritance is used.
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
      label(rootLabel),
      frozen(false),
      frozenUnique(false) {
    // size and tid set by operator new
  }

  /**
   * Special constructor for the root label.
   */
  Any(int) :
      sharedCount(0u),
      memoValueCount(0u),
      weakCount(1u),
      memoKeyCount(1u),
      label(nullptr),
      frozen(false),
      frozenUnique(false) {
    // size and tid set by operator new
  }

  /**
   * Copy constructor.
   */
  Any(const Any& o) : Any() {
    //
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
    auto ptr = static_cast<Any*>(allocate(size));
    ptr->size = static_cast<unsigned>(size);
    ptr->tid = get_thread_num();
    return ptr;
  }

  /**
   * Delete operator.
   */
  void operator delete(void* ptr) {
    auto o = static_cast<Any*>(ptr);
    o->destroy();
    o->deallocate();
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
   * Is this object reachable? An object is reachable if there exists a
   * shared, memo value, or weak pointer to it. This is equivalent to a weak
   * count of one or more. If only a memo key pointer to it exists, it is
   * not considered reachable, and it may be cleaned up during memo
   * maintenance.
   */
  bool isReachable() const {
    return numWeak() > 0u;
  }

  /**
   * Is there definitely only one pointer to this object? This is
   * conservative; it returns true if there is at most one shared or weak
   * pointer to the object. Note, in particular, that the presence of a memo
   * value pointer means that an unknown number of pointers (including zero
   * pointers) may update to the object in future.
   */
  bool isUnique() const {
    return numShared() <= 1u && numMemoValue() <= 1u && numWeak() <= 1u;
    // ^ recall first shared count increments memo value and weak counts
  }

  /**
   * Finalize. This is called when the memo value count reaches zero,
   * but before destruction and deallocation of the object. Object
   * resurrection is supported: if the finalizer results in a nonzero memo
   * value count, destruction and deallocation do not proceed.
   */
  virtual void finalize() {
    //
  }

  /**
   * Freeze the object.
   *
   * @param v Freeze visitor.
   */
  virtual void freeze_(const Freezer& v) = 0;

  /**
   * Copy the object.
   *
   * @param v Copy visitor.
   */
  virtual Any* copy_(const Copier& v) const = 0;

  /**
   * Recycle the object.
   *
   * @param v Recycle visitor.
   */
  virtual Any* recycle_(const Recycler& v) = 0;

  /**
   * Shared count.
   */
  unsigned numShared() const {
    return sharedCount.load();
  }

  /**
   * Increment the shared count.
   */
  void incShared() {
    if (++sharedCount == 1u) {
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
      releaseLabel();
      decMemoValue();
    }
  }

  /**
   * Memo value count.
   */
  unsigned numMemoValue() const {
    return memoValueCount.load();
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
      assert(numShared() == 0u);
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
   * Weak count.
   */
  unsigned numWeak() const {
    return weakCount.load();
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
      assert(numShared() == 0u);
      assert(numMemoValue() == 0u);
      decMemoKey();
    }
  }

  /**
   * Memo key count.
   */
  unsigned numMemoKey() const {
    return memoKeyCount.load();
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
      assert(numShared() == 0u);
      assert(numMemoValue() == 0u);
      assert(numWeak() == 0u);
      deallocate();
    }
  }

  /**
   * Get the label assigned to the object.
   */
  Label* getLabel() const {
    return label;
  }

  /**
   * Set the label assigned to the object. The shared count must be zero, and
   * the current label the root label. Typically this is applied to the new
   * object immediately after a copy operation.
   */
  void setLabel(Label* label) {
    assert(getLabel() == rootLabel);
    assert(numShared() == 0u);
    this->label = label;
  }

  /**
   * Replaced the label assigned to the object. The shared count must be
   * greater than zero. Typically this is applied to an object immediately
   * after a recycle operation.
   */
  void replaceLabel(Label* label) {
    assert(numShared() > 0u);
    if (label != this->label) {
      releaseLabel();
      this->label = label;
      holdLabel();
    }
  }

  /**
   * Freeze the object.
   *
   * @return Was the object *not* already frozen?
   */
  bool freeze() {
    bool frozenAlready = frozen;
    frozen = true;
    if (!frozenAlready) {
      frozenUnique = isUnique();
    }
    return !frozenAlready;
  }

  /**
   * Thaw the object.
   */
  void thaw() {
    frozen = false;
    frozenUnique = false;
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
   * Is the object frozen, and at the time of freezing, was there only one
   * pointer to it?
   */
  bool isFrozenUnique() const {
    return frozenUnique;
  }

protected:
  /**
   * Accept a visitor across member variables.
   */
  template<class Visitor>
  void accept_(const Visitor& v) {
    //
  }

private:
  /**
   * Increment the shared count of the label (if not null).
   */
  void holdLabel();

  /**
   * Decrement the shared count of the label (if not null). This is used
   * when the shared count for the object reduces to zero, while the memo
   * value count may still be greater than zero, in order to break any
   * reference cycles between objects and memos with the same label.
   */
  void releaseLabel();

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
  Label* label;

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
  int tid:30;

  /**
   * Is this frozen (read-only)?
   */
  bool frozen:1;

  /**
   * Is the object frozen, and at the time of freezing, was there only one
   * pointer to it?
   */
  bool frozenUnique:1;
};
}
