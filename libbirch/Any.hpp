/**
 * @file
 */
#pragma once

#include "libbirch/external.hpp"
#include "libbirch/memory.hpp"
#include "libbirch/Atomic.hpp"

namespace libbirch {
class Label;
class Finisher;
class Freezer;
class Copier;
class Recycler;
class Discarder;
class Restorer;

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
 * Reference-counted objects in LibBirch use four counts, rather than
 * the usual two (shared and weak), in order to support lazy deep copy
 * operations. These are:
 *
 *   - a *shared* count,
 *   - a *memo shared* count,
 *   - a *weak* count, and
 *   - a *memo weak* count.
 *
 * The shared and weak counts behave as normal. The memo shared and memo weak
 * counts serve to determine when an object is only reachable via a memo used
 * to bookkeep lazy deep copy operations. Objects that are only reachable via
 * a memo may be eligible for collection.
 *
 * The movement of the four counts triggers the following operations:
 *
 *   -# When the shared count reaches zero, the object is *discarded*.
 *   -# If the shared count subsequently becomes nonzero again (this is
 *      allowed as long as the memo shared count is nonzero), the object is
 *      *restored*.
 *   -# If the shared and memo shared counts reach 0, the object is
 *      *destroyed*.
 *   -# If the weak and memo weak counts reach 0, the object is *deallocated*.
 *
 * These operations behave as follows:
 *
 *   - **Discard** downgrades all shared pointers in the object to memo
 *     shared pointers.
 *   - **Restore** upgrades those memo shared pointers to shared pointers
 *     again.
 *   - **Destroy** calls the destructor of the object.
 *   - **Deallocate** collects the memory allocated to the object.
 *
 * The discard operation may be skipped when the destroy operation would
 * immediately follow anyway; i.e. when the shared count reaches zero and the
 * memo shared count is already at zero.
 *
 * At a high level, shared and weak pointers serve to determine which objects
 * are reachable from the user program, while memo shared and memo weak
 * pointers are used to determine which objects are reachable via a memo,
 * and to break reference cycles that are induced by the memo, but which
 * do not exist in the user program *per se*.
 *
 * A discarded object is in a state where reference cycles induced by a memo
 * are broken, but it is otherwise still a valid object, and may still be
 * accessible via a weak pointer in the user program.
 */
class Any {
public:
  using class_type_ = Any;
  using this_type_ = Any;

  /**
   * Constructor.
   */
  Any() :
      sharedCount(1u),
      memoSharedCount(1u),
      weakCount(1u),
      memoWeakCount(1u),
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
      sharedCount(1u),
      memoSharedCount(1u),
      weakCount(1u),
      memoWeakCount(1u),
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
    assert(memoSharedCount.load() == 0u);
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
   * shared, memo shared, or weak pointer to it. If only a memo weak pointer
   * to it exists, it is not considered reachable, and it may be cleaned up
   * during memo maintenance.
   */
  bool isReachable() const {
    return numWeak() > 0u;
  }

  /**
   * Is this object destroyed? An object is destroyed if there only exist
   * weak pointers to it.
   */
  bool isDestroyed() const {
    return numShared() == 0u && numMemoShared() == 0u;
  }

  /**
   * Is there definitely only one pointer to this object? This is
   * conservative; it returns true if there is at most one shared or weak
   * pointer to the object. Note, in particular, that the presence of a memo
   * shared pointer means that an unknown number of pointers (including zero
   * pointers) may update to the object in future.
   */
  bool isUnique() const {
    auto sharedCount = numShared();
    auto memoSharedCount = numMemoShared();
    auto weakCount = numWeak();
    auto memoWeakCount = numMemoWeak();

    return (sharedCount == 1u && memoSharedCount == 1u && weakCount == 1u &&
        memoWeakCount == 1u) || (sharedCount == 0u && memoSharedCount == 0u &&
        weakCount == 1u && memoWeakCount == 1u);
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
    assert(isFinished());
    auto old = packed.maskOr(FROZEN) & FROZEN;
    if (!old) {
      if (isUnique()) {
        packed.maskOr(FROZEN_UNIQUE);
      }
      freeze_();
    }
  }

  /**
   * Discard the object.
   */
  void discard() {
    auto old = packed.maskOr(DISCARDED) & DISCARDED;
    if (!old) {
      discard_();
    }
  }

  /**
   * Restore the object.
   */
  void restore() {
    auto old = packed.maskAnd(~DISCARDED) & DISCARDED;
    if (old) {
      restore_();
    }
  }

  /**
   * Copy the object.
   *
   * @param label The new label.
   */
  Any* copy(Label* label) {
    auto o = copy_(label);
    o->sharedCount.set(1u);
    o->memoSharedCount.set(1u);
    o->weakCount.set(1u);
    o->memoWeakCount.set(1u);
    o->label = label;
    o->tid = get_thread_num();
    o->packed.set(0u);
    o->holdLabel();
    return o;
  }

  /**
   * Recycle the object.
   *
   * @param label The new label.
   */
  Any* recycle(Label* label) {
    auto o = recycle_(label);
    o->packed.set(INITIALIZED);
    o->replaceLabel(label);
    return this;
  }

  /**
   * Shared count.
   */
  unsigned numShared() const {
    return sharedCount.load();
  }

  /**
   * Increment the shared count, possibly for the first time. This should be
   * used when a raw pointer is used to initialize a shared pointer,
   * otherwise incShared() is adequate. A check is made as to whether this is
   * the first shared pointer to the object, in which case behavior is
   * slightly different, and optimized.
   */
  void initShared() {
    auto initialized = packed.maskOr(INITIALIZED) & INITIALIZED;
    if (!initialized) {
      /* first shared pointer to this object; nothing to do, not even to
       * increment the shared count, as the object is initialized for this
       * purpose already */
    } else {
      /* not the first shared pointer to this object, behave as normal */
      incShared();
    }
  }

  /**
   * Increment the shared count. This can be used when a shared pointer to
   * the object is already known to exist, otherwise use initShared().
   */
  void incShared() {
    assert(isInitialized());
    if (++sharedCount == 1u) {
      incMemoShared();
      holdLabel();
      //discardLock.enterLeft();
      restore();
      //discardLock.exitLeft();
    }
  }

  /**
   * Decrement the shared count.
   */
  void decShared() {
    assert(numShared() > 0u);
    if (--sharedCount == 0u) {
      /* the sequence of operations to perform here is discard(), then
       * releaseLabel(), then decMemoShared(), but the first of these can be
       * expensive; consequently we skip it if possible, but for thread
       * safety hold an extra memo shared reference throughout so that the
       * memo shared count cannot be reduced to zero by another thread, thus
       * destroying the object, while we're still operating on it */
      incMemoShared();
      if (--memoSharedCount > 1u) {
        //discardLock.enterRight();
        discard();
        //discardLock.exitRight();
      } else {
        /* skip-discard optimization; no need to run discard as object is
         * about to be destroyed anyway */
      }
      releaseLabel();
      decMemoShared();
    }
  }

  /**
   * Memo shared count.
   */
  unsigned numMemoShared() const {
    return memoSharedCount.load();
  }

  /**
   * Increment the memo shared count.
   */
  void incMemoShared() {
    assert(numWeak() > 0u);
    memoSharedCount.increment();
  }

  /**
   * Decrement the memo shared count.
   */
  void decMemoShared() {
    assert(numMemoShared() > 0u);
    if (--memoSharedCount == 0u) {
      assert(numShared() == 0u);
      destroy();
      decWeak();
    }
  }

  /**
   * Simultaneously decrement the shared count and increment the shared memo
   * count.
   */
  void discardShared() {
    assert(numShared() > 0u);
    incMemoShared();
    if (--sharedCount == 0u) {
      discard();
      releaseLabel();
      decMemoShared();
    }
  }

  /**
   * Simultaneously increment the shared count and decrement the shared memo
   * count.
   */
  void restoreShared() {
    if (++sharedCount == 1u) {
      incMemoShared();
      holdLabel();
      restore();
    }
    decMemoShared();
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
    assert(numMemoWeak() > 0u);
    weakCount.increment();
  }

  /**
   * Decrement the weak count.
   */
  void decWeak() {
    assert(weakCount.load() > 0u);
    if (--weakCount == 0u) {
      assert(numShared() == 0u);
      assert(numMemoShared() == 0u);
      decMemoWeak();
    }
  }

  /**
   * Memo weak count.
   */
  unsigned numMemoWeak() const {
    return memoWeakCount.load();
  }

  /**
   * Increment the memo weak count.
   */
  void incMemoWeak() {
    memoWeakCount.increment();
  }

  /**
   * Decrement the memo weak count.
   */
  void decMemoWeak() {
    assert(memoWeakCount.load() > 0u);
    if (--memoWeakCount == 0u) {
      assert(numShared() == 0u);
      assert(numMemoShared() == 0u);
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
   * Is the object initialized?
   */
  bool isInitialized() const {
    return packed.load() & INITIALIZED;
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

  /**
   * Is the object discarded?
   */
  bool isDiscarded() const {
    return packed.load() & DISCARDED;
  }

protected:
  /**
   * Increment the shared count of the label. This is used during
   * initialization and restoration.
   */
  void holdLabel();

  /**
   * Decrement the shared count of the label. This is used during discard.
   */
  void releaseLabel();

  /**
   * Replace the label.
   */
  void replaceLabel(Label* label);

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
   * Discard the object.
   */
  virtual void discard_() = 0;

  /**
   * Restore the object.
   */
  virtual void restore_() = 0;

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
    assert(memoSharedCount.load() == 0u);
    this->size = size_();
    this->~Any();
  }

  /**
   * Deallocate the object. It should have previously been destroyed.
   */
  void deallocate() {
    assert(sharedCount.load() == 0u);
    assert(memoSharedCount.load() == 0u);
    assert(weakCount.load() == 0u);
    assert(memoWeakCount.load() == 0u);
    libbirch::deallocate(this, (unsigned)size, tid);
  }

  /**
   * Shared count.
   */
  Atomic<unsigned> sharedCount;

  /**
   * Memo shared count, or, if the shared count is nonzero, one plus the memo
   * shared count.
   */
  Atomic<unsigned> memoSharedCount;

  /**
   * Weak count, or, if the memo shared count is nonzero, one plus the weak
   * count.
   */
  Atomic<unsigned> weakCount;

  /**
   * Memo weak count, or, if the weak count is nonzero, one plus the memo
   * weak count.
   */
  Atomic<unsigned> memoWeakCount;

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
   *   - *initialized* flag,
   *   - *finished* flag,
   *   - *frozen* flag,
   *   - *frozen unique* flag (is the object frozen, and at the time of
   *     freezing, was there only one pointer to it?), and
   *   - *discarded* flag.
   *
   * Each occupies 1 bit, from the right.
   */
  Atomic<uint16_t> packed;

  /*
   * Flags for packed.
   */
  enum {
    INITIALIZED = 1,
    FINISHED = 2,
    FROZEN = 4,
    FROZEN_UNIQUE = 8,
    DISCARDED = 16
  };
};
}
