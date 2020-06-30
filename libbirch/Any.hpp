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
 * Base class providing reference counting, cycle breaking, and lazy deep
 * copy support.
 *
 * @ingroup libbirch
 *
 * @attention A newly created object of type Any, or of a type derived from
 * it, must be assigned to at least one Shared pointer in its lifetime to
 * be correctly destroyed and deallocated. Furthermore, in order to work
 * correctly with multiple inheritance, Any must be the *first* base class.
 */
class Any {
public:
  using class_type_ = Any;
  using this_type_ = Any;

  /**
   * Constructor.
   */
  Any() :
      label(root_label),
      sharedCount(0u),
      weakCount(1u),
      memoCount(1u),
      size(0u),
      tid(get_thread_num()),
      flags(0u),
      color(BLACK) {
    //
  }

  /**
   * Special constructor for the root label.
   */
  Any(int) :
      label(nullptr),
      sharedCount(0u),
      weakCount(1u),
      memoCount(1u),
      size(0u),
      tid(0),
      flags(0u),
      color(BLACK) {
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
   * Get the class name.
   */
  virtual bi::type::String getClassName() const {
    return "Any";
  }

  /**
   * Finish the object.
   */
  void finish(Label* label) {
    auto old = flags.maskOr(FINISHED) & FINISHED;
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
      auto old = flags.maskOr(FROZEN) & FROZEN;
      if (!old) {
        if (isUnique()) {
          flags.maskOr(FROZEN_UNIQUE);
        }
        freeze_();
      }
    }
  }

  /**
   * Thaw the object.
   */
  void thaw() {
    flags.store(0u);
  }

  /**
   * Copy the object.
   *
   * @param label The new label.
   */
  Any* copy(Label* label) {
    auto o = copy_(label);
    o->label = label;
    o->sharedCount.set(0u);
    o->weakCount.set(1u);
    o->memoCount.set(1u);
    o->size = 0u;
    o->tid = get_thread_num();
    o->flags.set(0u);
    return o;
  }

  /**
   * Recycle the object. This can be used as an optimization in place of
   * copy(), where only one pointer remains to the source object, and it would
   * otherwise be copied and then destroyed; instead, the source object is
   * modified for reuse.
   *
   * @param label The new label.
   */
  Any* recycle(Label* label) {
    auto o = recycle_(label);
    o->label = label;
    return this;
  }

  /**
   * Mark the object.
   *
   * This performs the `MarkGray()` operation of @ref Bacon2001
   * "Bacon & Rajan (2001)".
   */
  void mark() {
    if (color.exchange(GRAY) != GRAY) {
      mark_();
    }
  }

  /**
   * Scan the object.
   *
   * @param reachable Is the object definitely reachable?
   *
   * This performs the `Scan()` (when @p reachable is false) and `ScanBlack()`
   * (when @p reachable is true) operations of @ref Bacon2001
   * "Bacon & Rajan (2001)".
   */
  void scan(const bool reachable) {
    if (reachable || numShared() > 0u) {
      if (color.exchange(BLACK) != BLACK) {
        scan_(true);
      }
    } else {
      if (color.exchange(WHITE) != WHITE) {
        scan_(false);
      }
    }
  }

  /**
   * Collect the object.
   *
   * This performs the `CollectWhite()` operation of @ref Bacon2001
   * "Bacon & Rajan (2001)".
   */
  void collect() {
    if (color.exchange(BLACK) != BLACK) {
      collect_();
      destroy();
      decWeak();
    }
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

    /* if the count will reduce to nonzero, this is possible the root of
     * a cycle; check this before decrementing rather than after, as otherwise
     * another thread may destroy the object while this thread registers */
    if (sharedCount.load() > 1u && color.exchange(PURPLE) == BLACK) {
      register_possible_root(this);
    }

    /* decrement */
    if (--sharedCount == 0u) {
      destroy();
      decWeak();
    }
  }

  /**
   * Decrement the shared count during mark() operation.
   */
  void breakShared() {
    assert(numShared() > 0u);
    sharedCount.decrement();
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
   * Memo count.
   */
  unsigned numMemo() const {
    return memoCount.load();
  }

  /**
   * Increment the memo count.
   */
  void incMemo() {
    memoCount.increment();
  }

  /**
   * Decrement the memo count.
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
   * Is there only one shared or weak pointer to this object?
   */
  bool isUnique() const {
    return numShared() <= 1u && numWeak() <= 1u;
  }

  /**
   * Is the object finished?
   */
  bool isFinished() const {
    return flags.load() & FINISHED;
  }

  /**
   * Is the object frozen?
   */
  bool isFrozen() const {
    return flags.load() & FROZEN;
  }

  /**
   * Is the object frozen, and at the time of freezing, was there only one
   * pointer to it?
   */
  bool isFrozenUnique() const {
    return flags.load() & FROZEN_UNIQUE;
  }

  /**
   * Is the object colored black?
   */
  bool isBlack() const {
    return color.load() == BLACK;
  }

  /**
   * Set the color to black.
   */
  void setBlack() {
    color.store(BLACK);
  }

  /**
   * Is the object colored purple?
   */
  bool isPurple() const {
    return color.load() == PURPLE;
  }

  /**
   * Set the color to purple.
   */
  void setPurple() {
    color.store(PURPLE);
  }

  /**
   * Is the object colored white?
   */
  bool isWhite() const {
    return color.load() == WHITE;
  }

  /**
   * Set the color to white.
   */
  void setWhite() {
    color.store(WHITE);
  }

  /**
   * Is the object colored gray?
   */
  bool isGray() const {
    return color.load() == GRAY;
  }

  /**
   * Set the color to gray.
   */
  void setGray() {
    color.store(GRAY);
  }

protected:
  /**
   * Called internally by finish() to recurse into member variables.
   */
  virtual void finish_(Label* label) = 0;

  /**
   * Called internally by freeze() to recurse into member variables.
   */
  virtual void freeze_() = 0;

  /**
   * Called internally by copy() to ensure the most derived type is copied.
   */
  virtual Any* copy_(Label* label) const = 0;

  /**
   * Called internally by recycle() to ensure the most derived type is
   * recycled.
   */
  virtual Any* recycle_(Label* label) = 0;

  /**
   * Called internally by mark() to recurse into member variables.
   */
  virtual void mark_() = 0;

  /**
   * Called internally by scan() to recurse into member variables.
   */
  virtual void scan_(const bool reachable) = 0;

  /**
   * Called internally by collect() to recurse into member variables.
   */
  virtual void collect_() = 0;

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
    libbirch::deallocate(this, size, tid);
  }

  /**
   * Label of the object.
   */
  Label* label;

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
   * Size of the object. This is initially set to zero. Upon destruction, it
   * is set to the correct size with a virtual function call.
   */
  unsigned size;

  /**
   * Id of the thread associated with the object. This is set immediately
   * after allocation. It is used to return the allocation to the correct
   * pool after use, even when returned by a different thread.
   */
  int tid;

  /**
   * Bitfield containing, from right to left:
   *
   *   - *finished* flag,
   *   - *frozen* flag,
   *   - *frozen unique* flag.
   *
   * These first three used for lazy deep copy operations as in
   * @ref Murray2020 "Murray (2020)".
   */
  Atomic<uint8_t> flags;

  /**
   * Flags.
   */
  enum {
    FINISHED = 1 << 0,
    FROZEN = 1 << 1,
    FROZEN_UNIQUE = 1 << 2
  };

  /**
   * Color.
   *
   * This is used for cycle collection as in @ref Bacon2001
   * "Bacon & Rajan (2001)", with some adaptation. A small difference is
   * that we do not allow an object colored purple to return to black, as it
   * creates a race condition in our implementation. This only has the effect
   * of potentially (wastefully) checking for cycles from a possible root that
   * is definitely reachable. It also means that the *buffered* flag in
   * @ref Bacon2001 "Bacon & Rajan (2001)" is unnecessary: the color purple
   * suffices to indicate that an object has been registered as a possible
   * root.
   */
  Atomic<uint8_t> color;

  /**
   * Colors. These are the 2-bit values described, but shifted
   * to the position of the color field in flags.
   */
  enum {
    BLACK = 0,
    PURPLE = 1,
    GRAY = 2,
    WHITE = 3
  };
};
}
