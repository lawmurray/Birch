/**
 * @file
 */
#pragma once

#include "libbirch/external.hpp"
#include "libbirch/memory.hpp"
#include "libbirch/Atomic.hpp"

namespace libbirch {
class Label;

/**
 * Base class for reference counted objects.
 *
 * @ingroup libbirch
 *
 * @attention A newly created object of type Any, or of a type derived from
 * it, must be assigned to at least one Shared pointer in its lifetime to
 * be correctly destroyed and deallocated. Furthermore, in order to work
 * correctly with multiple inheritance, Any must be the *first* base class.
 *
 * Reference-counted objects in LibBirch use three counts, rather than
 * the usual two (shared and weak), in order to support lazy deep copy
 * operations. These are:
 *
 *   - a *shared* count,
 *   - a *weak* count, and
 *   - a *memo* count.
 *
 * The shared and weak counts behave as normal. The memo count is used for
 * keys in the memos used to bookkeep lazy deep copy operations.
 *
 * The movement of the three counts triggers the following operations:
 *
 *   -# When the shared count reaches zero, the object is *destroyed*.
 *   -# If the weak and memo weak counts reach 0, the object is *deallocated*.
 *
 * Shared and weak pointers determine which objects are reachable from the
 * user program, while memo pointers are used to determine which objects are
 * reachable via memos. Memos voluntarily surrender their pointers during
 * cleanup, upon discovery that there are no shared or weak pointers to those
 * referents.
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
      weakCount(1u),
      memoCount(1u),
      label(rootLabel),
      tid(get_thread_num()),
      size(0u),
      packed(0u) {
    //
  }

  /**
   * Special constructor for the root label.
   */
  Any(int) :
      sharedCount(0u),
      weakCount(1u),
      memoCount(1u),
      label(nullptr),
      tid(0),
      size(0u),
      packed(0u) {
    //
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
  }

  /**
   * New operator.
   */
  void* operator new(std::size_t size) {
    return allocate(size);
  }

  /**
   * Delete operator.
   */
  void operator delete(void* ptr) {
    auto o = static_cast<Any*>(ptr);
    o->deallocate();
  }

  /**
   * Assignment operator.
   */
  Any& operator=(const Any&) {
    return *this;
  }

  /**
   * Is this object reachable? An object is reachable if there exists a
   * shared or weak pointer to it. If only memo pointers to it exists, it is
   * not considered reachable, and it may be cleaned up during memo
   * maintenance.
   */
  bool isReachable() const {
    return numWeak() > 0u;
  }

  /**
   * Is this object destroyed? An object is destroyed if there only exist
   * weak pointers to it.
   */
  bool isDestroyed() const {
    return numShared() == 0u;
  }

  /**
   * Is there only one pointer to this object?
   */
  bool isUnique() const {
    auto sharedCount = numShared();
    auto weakCount = numWeak();
    auto memoCount = numMemo();

    return (sharedCount == 1u && weakCount == 1u && memoCount == 1u) ||
        (sharedCount == 0u && weakCount == 1u && memoCount == 1u);
  }

  /**
   * Get the class name.
   */
  virtual bi::type::String getClassName() const {
    return "Any";
  }

  /**
   * Finish the object.
   */
  void finish(Label* label) {
    auto old = packed.maskOr(FINISHED) & FINISHED;
    if (!old) {
      finish_(label);
    }
  }

  /**
   * Freeze the object.
   */
  void freeze() {
    /* objects must be finished before they are frozen; however, it can be the
     * case that, after one thread reaches an object during its finish pass,
     * but before it reaches it again during its freeze pass, a second thread
     * freezes and subsequently copies it; the first thread then reaches the
     * new copy during its freeze pass; it should not attempt to finish or
     * freeze such an object---the work has already been done by the second
     * thread */
    if (isFinished()) {
      auto old = packed.maskOr(FROZEN) & FROZEN;
      if (!old) {
        if (isUnique()) {
          packed.maskOr(FROZEN_UNIQUE);
        }
        freeze_();
      }
    }
  }

  /**
   * Thaw the object.
   */
  void thaw() {
    packed.store(0u);
  }

  /**
   * Copy the object.
   *
   * @param label The new label.
   */
  Any* copy(Label* label) {
    auto o = copy_(label);
    o->sharedCount.set(0u);
    o->weakCount.set(1u);
    o->memoCount.set(1u);
    o->label = label;
    o->tid = get_thread_num();
    o->size = 0u;
    o->packed.set(0u);
    return o;
  }

  /**
   * Recycle the object.
   *
   * @param label The new label.
   */
  Any* recycle(Label* label) {
    auto o = recycle_(label);
    o->label = label;
    return this;
  }

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
    sharedCount.increment();
  }

  /**
   * Decrement the shared count.
   */
  void decShared() {
    assert(numShared() > 0u);
    if (--sharedCount == 0u) {
      destroy();
      decWeak();
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
    assert(numMemo() > 0u);
    weakCount.increment();
  }

  /**
   * Decrement the weak count.
   */
  void decWeak() {
    assert(weakCount.load() > 0u);
    if (--weakCount == 0u) {
      assert(numShared() == 0u);
      decMemo();
    }
  }

  /**
   * Memo weak count.
   */
  unsigned numMemo() const {
    return memoCount.load();
  }

  /**
   * Increment the memo weak count.
   */
  void incMemo() {
    memoCount.increment();
  }

  /**
   * Decrement the memo weak count.
   */
  void decMemo() {
    assert(memoCount.load() > 0u);
    if (--memoCount == 0u) {
      assert(numShared() == 0u);
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
   * Is the object finished?
   */
  bool isFinished() const {
    return packed.load() & FINISHED;
  }

  /**
   * Is the object frozen?
   */
  bool isFrozen() const {
    return packed.load() & FROZEN;
  }

  /**
   * Is the object frozen, and at the time of freezing, was there only one
   * pointer to it?
   */
  bool isFrozenUnique() const {
    return packed.load() & FROZEN_UNIQUE;
  }

protected:
  /**
   * Finish the member variables of the object.
   */
  virtual void finish_(Label* label) = 0;

  /**
   * Freeze the member variables of the object.
   */
  virtual void freeze_() = 0;

  /**
   * Copy the object.
   *
   * @param label The new label.
   */
  virtual Any* copy_(Label* label) const = 0;

  /**
   * Recycle the object.
   *
   * @param label The new label.
   */
  virtual Any* recycle_(Label* label) = 0;

  /**
   * Size of the object.
   */
  virtual uint16_t size_() const {
    return (uint16_t)sizeof(*this);
  }

  /**
   * Accept a visitor across member variables.
   */
  template<class Visitor>
  void accept_(const Visitor& v) {
    //
  }

private:
  /**
   * Destroy, but do not deallocate, the object.
   */
  void destroy() {
    assert(sharedCount.load() == 0u);
    this->size = size_();
    this->~Any();
  }

  /**
   * Deallocate the object. It should have previously been destroyed.
   */
  void deallocate() {
    assert(sharedCount.load() == 0u);
    assert(weakCount.load() == 0u);
    assert(memoCount.load() == 0u);
    libbirch::deallocate(this, (unsigned)size, tid);
  }

  /**
   * Shared count.
   */
  Atomic<unsigned> sharedCount;

  /**
   * Weak count, or, if the shared count is nonzero, one plus the weak count.
   */
  Atomic<unsigned> weakCount;

  /**
   * Memo count, or, if the weak count is nonzero, one plus the memo count.
   */
  Atomic<unsigned> memoCount;

  /**
   * Label of the object.
   */
  Label* label;

  /**
   * Id of the thread associated with the object. This is set immediately
   * after allocation. It is used to return the allocation to the correct
   * pool after use, even when returned by a different thread.
   */
  int tid;

  /**
   * Size of the object. This is initially set to zero. Upon destruction, it
   * is set to the correct size with a virtual function call.
   */
  uint16_t size;

  /**
   * Bitfield containing:
   *
   *   - *finished* flag,
   *   - *frozen* flag,
   *   - *frozen unique* flag (is the object frozen, and at the time of
   *     freezing, was there only one pointer to it?).
   *
   * Each occupies 1 bit, from the right.
   */
  Atomic<uint16_t> packed;

  /*
   * Flags for packed.
   */
  enum {
    FINISHED = 1,
    FROZEN = 2,
    FROZEN_UNIQUE = 4
  };
};
}
