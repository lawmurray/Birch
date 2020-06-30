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
      sharedCount(0u),
      weakCount(1u),
      memoCount(1u),
      markCount(0u),
      label(root_label),
      tid(get_thread_num()),
      size(0u),
      flags(0u) {
    //
  }

  /**
   * Special constructor for the root label.
   */
  Any(int) :
      sharedCount(0u),
      weakCount(1u),
      memoCount(1u),
      markCount(0u),
      label(nullptr),
      tid(0),
      size(0u),
      flags(0u) {
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
    o->sharedCount.set(0u);
    o->weakCount.set(1u);
    o->memoCount.set(1u);
    o->markCount.set(0u);
    o->label = label;
    o->tid = get_thread_num();
    o->size = 0u;
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
    mark_();
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
    scan_(reachable);
  }

  /**
   * Collect the object.
   *
   * This performs the `CollectWhite()` operation of @ref Bacon2001
   * "Bacon & Rajan (2001)".
   */
  void collect() {
    collect_();
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
    } else if (color.exchange(PURPLE) != PURPLE) {
      register_possible_root(this);
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
   * Mark count.
   */
  unsigned numMark() const {
    return markCount.load();
  }

  /**
   * Increment the mark count.
   */
  void incMark() {
    markCount.increment();
  }

  /**
   * Clear the mark count back to zero.
   */
  void clearMark() {
    markCount.store(0u);
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
   * Mark count. This is used during the cycle collection algorithm to count
   * the number times the object is reached via a shared pointer when marking
   * from the root set. This is slightly different to @ref Bacon2001
   * "Bacon & Rajan (2001)": where they decrement the shared reference count
   * and test the result against zero, we increment this mark count and
   * compare it against the shared count, so as to leave the latter untouched.
   */
  Atomic<unsigned> markCount;

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
   * "Bacon & Rajan (2001)". The colors are encoded as:
   *
   *   - `00` black,
   *   - `01` purple,
   *   - `10` white,
   *   - `11` gray.
   *
   * These values are deliberately chosen given the state diagram in Figure 3
   * of @ref Bacon2001 "Bacon & Rajan (2001)". They allow the convenience of
   * atomic mask operations to be used to transition to each color from each
   * other (valid) color. In particular:
   *
   *   - `|01` transitions to purple (from black),
   *   - `|11` transitions to gray (from black or purple),
   *   - `&10` transitions to white (from gray)
   *   - `&00` transitions to black (from purple, gray or white).
   *
   * However, the transition from purple to black is not used due to a race
   * condition. This only has the effect of potentially (wastefully) checking
   * for cycles from a candidate root node that may otherwise have been
   * eliminated earlier. But it also means that the *buffered* flag in
   * @ref Bacon2001 "Bacon & Rajan (2001)" is unnecessary: the color purple
   * suffices to indicate that an object is in the root set.
   */
  Atomic<uint8_t> color;

  /**
   * Colors. These are the 2-bit values described, but shifted
   * to the position of the color field in flags.
   */
  enum {
    BLACK = 0,
    PURPLE = 1,
    WHITE = 2,
    GRAY = 3
  };
};
}
