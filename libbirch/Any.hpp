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
      markCount(0u),
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
    o->markCount.set(0u);
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
   * Is the object buffered as a potential root of a cycle?
   */
  bool isBuffered() const {
    return packed.load() & BUFFERED;
  }

  /**
   * Set the buffered flag.
   */
  void setBuffered() {
    packed.maskOr(BUFFERED);
  }

  /**
   * Clear the buffered flag.
   */
  void clearBuffered() {
    packed.maskAnd(~BUFFERED);
  }

  /**
   * Is the object colored black?
   */
  bool isBlack() const {
    return (packed.load() & COLOR) == BLACK;
  }

  /**
   * Set the color to black.
   */
  void setBlack() {
    packed.maskAnd(~COLOR | BLACK);
  }

  /**
   * Is the object colored purple?
   */
  bool isPurple() const {
    return (packed.load() & COLOR) == PURPLE;
  }

  /**
   * Set the color to purple.
   */
  void setPurple() {
    [[maybe_unused]] auto old = packed.maskOr(PURPLE);
    assert((old & COLOR) == BLACK);
  }

  /**
   * Is the object colored white?
   */
  bool isWhite() const {
    return (packed.load() & COLOR) == WHITE;
  }

  /**
   * Set the color to white.
   */
  void setWhite() {
    [[maybe_unused]] auto old = packed.maskAnd(~COLOR|WHITE);
    assert((old & COLOR) == GRAY);
  }

  /**
   * Is the object colored gray?
   */
  bool isGray() const {
    return (packed.load() & COLOR) == GRAY;
  }

  /**
   * Set the color to gray.
   */
  void setGray() {
    [[maybe_unused]] auto old = packed.maskOr(GRAY);
    assert((old & COLOR) == BLACK || (old & COLOR) == PURPLE);
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
   *   - *frozen unique* flag (is the object frozen, and at the time of
   *     freezing, was there only one pointer to it?)
   *
   * these first three used for lazy deep copy operations as in
   * @ref Murray2020 "Murray (2020)", then
   *
   *   - *buffered* flag, and
   *   - a 2-bit *color*,
   *
   * where these next two are used for cycle collection as in
   * @ref Bacon2001 "Bacon & Rajan (2001)". The colors are encoded as:
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
   *   - `| 01` transitions to purple (from black),
   *   - `| 11` transitions to gray (from black or purple),
   *   - `& 10` transitions to white (from gray)
   *   - `& 00` transitions to black (from purple, gray or white).
   */
  Atomic<uint16_t> packed;

  /**
   * Flags for packed.
   */
  enum {
    FINISHED = 1 << 0,
    FROZEN = 1 << 1,
    FROZEN_UNIQUE = 1 << 2,
    BUFFERED = 1 << 3,
    COLOR = (1 << 4) | (1 << 5)
  };

  /**
   * Colors for packed. These are the 2-bit values described, but shifted
   * to the position of the color field in packed.
   */
  enum {
    BLACK = 0 << 4,
    PURPLE = 1 << 4,
    WHITE = 2 << 4,
    GRAY = 3 << 4
  };
};
}
