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
      memoCount(1u),
      size(0u),
      tid(get_thread_num()),
      flags(0u) {
    //
  }

  /**
   * Special constructor for the root label.
   */
  Any(int) :
      label(nullptr),
      sharedCount(0u),
      memoCount(1u),
      size(0u),
      tid(0),
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
    if (set(FINISHED)) {
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
    if (is(FINISHED)) {
      if (set(FROZEN)) {
        if (sharedCount.load() == 1u) {
          // ^ small optimization: isUnique() makes sense, but unnecessarily
          //   loads memoCount as well, which is unnecessary for a objects
          //   that are not frozen
          set(FROZEN_UNIQUE);
        }
        freeze_();
      }
    }
  }

  /**
   * Thaw the object.
   */
  void thaw() {
    unset(~BUFFERED);
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
    o->memoCount.set(1u);
    o->size.set(0u);
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
  void recycle(Label* label) {
    recycle_(label);
  }

  /**
   * Mark the object.
   *
   * This performs the `MarkGray()` operation of @ref Bacon2001
   * "Bacon & Rajan (2001)".
   */
  void mark() {
    if (set(MARKED)) {
      unset(BUFFERED|SCANNED|REACHED|COLLECTED);  // unset for later passes
      mark_();
    }
  }

  /**
   * Scan the object.
   *
   * This performs the `Scan()` operation of @ref Bacon2001
   * "Bacon & Rajan (2001)".
   */
  void scan() {
    if (set(SCANNED)) {
      unset(MARKED);  // unset for next time
      if (numShared() > 0u) {
        if (set(REACHED)) {
          reach_();
        }
      } else {
        scan_();
      }
    }
  }

  /**
   * Reach the object.
   *
   * This performs the `ScanBlack()` operation of @ref Bacon2001
   * "Bacon & Rajan (2001)".
   */
  void reach() {
    if (set(SCANNED)) {
      unset(MARKED);  // unset for next time
    }
    if (set(REACHED)) {
      reach_();
    }
  }

  /**
   * Collect the object.
   *
   * This performs the `CollectWhite()` operation of @ref Bacon2001
   * "Bacon & Rajan (2001)".
   */
  void collect() {
    if (set(COLLECTED) && !is(REACHED)) {
      register_unreachable(this);
      collect_();
    }
  }

  /**
   * Destroy, but do not deallocate, the object.
   */
  void destroy() {
    assert(sharedCount.load() == 0u);
    this->size.store(size_());
    this->~Any();
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
   * Decrement the shared count. This decrements the count; if the new count
   * is nonzero, it registers the object as a possible root for cycle
   * collection, or if the new count is zero, it destroys the object.
   */
  void decShared() {
    assert(numShared() > 0u);

    /* if the count will reduce to nonzero, this is possible the root of
     * a cycle; check this before decrementing rather than after, as otherwise
     * another thread may destroy the object while this thread registers */
    if (sharedCount.load() > 1u && set(BUFFERED)) {
      register_possible_root(this);
    }

    /* decrement */
    if (--sharedCount == 0u) {
      destroy();
      decMemo();
    }
  }

  /**
   * Decrement the shared count with a known acyclic referent. This decrements
   * the count, and if the new count is zero, it destroys the object. The
   * caller asserts that the object is of acyclic type (@see is_acyclic), so
   * that there is no need to register the object as a possible root for cycle
   * collection.
   */
  void decSharedAcyclic() {
    assert(numShared() > 0u);
    if (--sharedCount == 0u) {
      destroy();
      decMemo();
    }
  }

  /**
   * Decrement the shared count for an object that will remain reachable. The
   * caller asserts that the object will remain reachable after the operation.
   * The object will not be destroyed, and will not be registered as a
   * possible root for cycle collection.
   */
  void decSharedReachable() {
    assert(numShared() > 0u);
    sharedCount.decrement();
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
   * Has the object been destroyed?
   */
  bool isDestroyed() const {
    /* the shared count being zero is not a good indicator here: new objects
     * start with a zero count, and objects may temporarily have a zero count
     * during cycle collection; a better indicator is a nonzero size, which is
     * only set in destroy() */
    return size.load() > 0u;
  }

  /**
   * Is there only one pointer (of any type) to this object?
   */
  bool isUnique() const {
    return numShared() == 1u && numMemo() == 1u;
  }

  /**
   * Is the object frozen?
   */
  bool isFrozen() const {
    return is(FROZEN);
  }

  /**
   * Is the object frozen, and at the time of freezing, was there only one
   * shared pointer to it?
   */
  bool isFrozenUnique() const {
    return is(FROZEN_UNIQUE);
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
  virtual void recycle_(Label* label) {
    this->label = label;
  }

  /**
   * Called internally by mark() to recurse into member variables.
   */
  virtual void mark_() = 0;

  /**
   * Called internally by scan() to recurse into member variables.
   */
  virtual void scan_() = 0;

  /**
   * Called internally by reach() to recurse into member variables.
   */
  virtual void reach_() = 0;

  /**
   * Called internally by collect() to recurse into member variables.
   */
  virtual void collect_() = 0;

  /**
   * Size of the object.
   */
  virtual unsigned size_() const {
    return sizeof(*this);
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
   * Deallocate the object. It should have previously been destroyed.
   */
  void deallocate() {
    assert(sharedCount.load() == 0u);
    assert(memoCount.load() == 0u);
    libbirch::deallocate(this, size.load(), tid);
  }

  /**
   * Are one or more flags set?
   *
   * @param flags Bitmask of the flags (possibly just one).
   *
   * @return Are one or more of the flags set?
   */
  bool is(const uint8_t flags) const {
    return this->flags.load() & flags;
  }

  /**
   * Set one or more flags.
   *
   * @param flags Bitmask of the flags (possibly just one).
   *
   * @return Were one or more of the flags not set?
   */
  bool set(const uint8_t flags) {
    return !(this->flags.maskOr(flags) & flags);
  }

  /**
   * Unset one or more flags.
   *
   * @param flags Bitmask of the flags (possibly just one).
   *
   * @return Were one or more of the flags set?
   */
  bool unset(const uint8_t flags) {
    return this->flags.maskAnd(~flags) & flags;
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
   * Memo count, or, if the shared count is nonzero, one plus the memo count.
   */
  Atomic<unsigned> memoCount;

  /**
   * Size of the object. This is initially set to zero. Upon destruction, it
   * is set to the correct size with a virtual function call.
   */
  Atomic<unsigned> size;

  /**
   * Id of the thread associated with the object. This is set immediately
   * after allocation. It is used to return the allocation to the correct
   * pool after use, even when returned by a different thread.
   */
  int tid:24;

  /**
   * Bitfield containing flags. These are, from least to most significant
   * bits:
   *
   *   - *finished*,
   *   - *frozen*,
   *   - *single reference when frozen*---
   *
   * ---these used for lazy deep copy operations as in @ref Murray2020
   * "Murray (2020)"---then
   *
   *   - *buffered*,
   *   - *marked*,
   *   - *scanned*,
   *   - *reached*,
   *   - *collected*---
   *
   * ---these used for cycle collection as in @ref Bacon2001
   * "Bacon & Rajan (2001)".
   *
   * The second group of flags take the place of the colors described in
   * @ref Bacon2001 "Bacon & Rajan (2001)". The reason is to ensure that both
   * the bookkeeping required during normal execution can be multithreaded,
   * and likewise that the operations required during cycle collection can be
   * multithreaded. The basic principle to ensure this is that flags can be
   * safely set during normal execution (with atomic operations), but should
   * only be unset with careful consideration of thread safety.
   *
   * Notwithstanding, the flags do map to colors in @ref Bacon2001
   * "Bacon & Rajan (2001)":
   *
   *   - *buffered* maps to *purple* as well as the same-named *buffered*
   *     flag (the transition from purple to black is disallowed for
   *     reasons of thread safety, so purple and *buffered* are, anyway,
   *     synonymous),
   *   - *marked* maps to *gray*,
   *   - *scanned* and *reachable* together map to *black* (both on) or
   *     *white* (first on, second off),
   *   - *collected* is set once a white object has been destroyed.
   *
   * Disallowing the transition from purple to black has the effect of
   * potentially (and wastefully) checking for cycles from an object that
   * could otherwise have been eliminated as a possible root. This compromises
   * performance, but not correctness.. The use of these flags also resolves
   * some thread safety issues that can otherwise exist during the scan
   * operation, when coloring an object white (eligible for collection) then
   * later recoloring it black (reachable); the sequencing of this coloring
   * can become problematic with multiple threads.
   */
  Atomic<uint8_t> flags;

  /**
   * Flags.
   */
  enum Flag : uint8_t {
    BUFFERED = (1u << 0u),
    FINISHED = (1u << 1u),
    FROZEN = (1u << 2u),
    FROZEN_UNIQUE = (1u << 3u),
    MARKED = (1u << 4u),
    SCANNED = (1u << 5u),
    REACHED = (1u << 6u),
    COLLECTED = (1u << 7u)
  };
};
}
