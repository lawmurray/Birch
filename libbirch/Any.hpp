/**
 * @file
 */
#pragma once

#include "libbirch/external.hpp"
#include "libbirch/memory.hpp"
#include "libbirch/Init.hpp"
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
   * Finish the object.
   */
  void finish(Label* label) {
    if (!(flags.exchangeOr(FINISHED) & FINISHED)) {
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
    if (flags.load() & FINISHED) {
      if (!(flags.exchangeOr(FROZEN) & FROZEN)) {
        if (sharedCount.load() == 1u) {
          // ^ small optimization: isUnique() makes sense, but unnecessarily
          //   loads memoCount as well, which is unnecessary for a objects
          //   that are not frozen
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
    flags.maskAnd(~BUFFERED);
  }

  /**
   * Copy the object.
   *
   * @param label The new label.
   */
  Any* copy(Label* label) {
    auto o = copy_(label);
    new (&o->label) Init<Label>(label);
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
    this->label.replace(label);
    this->flags.maskAnd(~(FINISHED|FROZEN|FROZEN_UNIQUE));
    recycle_(label);
  }

  /**
   * Mark the object.
   *
   * This performs the `MarkGray()` operation of @ref Bacon2001
   * "Bacon & Rajan (2001)".
   */
  void mark() {
    if (!(flags.exchangeOr(MARKED) & MARKED)) {
      flags.maskAnd(~(BUFFERED|SCANNED|REACHED|COLLECTED));
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
    if (!(flags.exchangeOr(SCANNED) & SCANNED)) {
      flags.maskAnd(~MARKED);  // unset for next time
      if (numShared() > 0u) {
        if (!(flags.exchangeOr(REACHED) & REACHED)) {
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
    if (!(flags.exchangeOr(SCANNED) & SCANNED)) {
      flags.maskAnd(~MARKED);  // unset for next time
    }
    if (!(flags.exchangeOr(REACHED) & REACHED)) {
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
    auto old = flags.exchangeOr(COLLECTED);
    if (!(old & COLLECTED) && !(old & REACHED)) {
      register_unreachable(this);
      collect_();
    }
  }

  /**
   * Destroy, but do not deallocate, the object.
   */
  void destroy() {
    assert(sharedCount.load() == 0u);
    this->flags.maskAnd(~POSSIBLE_ROOT);
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
    flags.maskAnd(~POSSIBLE_ROOT);
    // ^ any interleaving with decShared() switching on POSSIBLE_ROOT should
    //   not be problematic; having it on is never a correctness issue, only
    //   a performance issue, and as long as one thread can reach the object
    //   it is fine to be off
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
    if (sharedCount.load() > 1u &&
        !(flags.exchangeOr(BUFFERED|POSSIBLE_ROOT) & BUFFERED)) {
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
   *
   * Acyclic objects occur in @ref Bacon2001 "Bacon & Rajan (2001)", where
   * they are colored *green*.
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
    return label.get();
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
   * Is this object the possible root of a cycle?
   */
  bool isPossibleRoot() const {
    return flags.load() & POSSIBLE_ROOT;
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
    return flags.load() & FROZEN;
  }

  /**
   * Is the object frozen, and at the time of freezing, was there only one
   * shared pointer to it?
   */
  bool isFrozenUnique() const {
    return flags.load() & FROZEN_UNIQUE;
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
   * Label of the object.
   */
  Init<Label> label;

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
  int16_t tid;

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
   *   - *possible root*,
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
   *   - *possible root* maps to *purple*,
   *   - *marked* maps to *gray*,
   *   - *scanned* and *reachable* together map to *black* (both on) or
   *     *white* (first on, second off),
   *   - *collected* is set once a white object has been destroyed.
   *
   * The use of these flags also resolves some thread safety issues that can
   * otherwise exist during the scan operation, when coloring an object white
   * (eligible for collection) then later recoloring it black (reachable); the
   * sequencing of this coloring can become problematic with multiple threads.
   */
  Atomic<uint16_t> flags;

  /**
   * Flags.
   */
  enum Flag : uint16_t {
    FINISHED = (1u << 0u),
    FROZEN = (1u << 1u),
    FROZEN_UNIQUE = (1u << 2u),
    POSSIBLE_ROOT = (1u << 3u),
    BUFFERED = (1u << 4u),
    MARKED = (1u << 5u),
    SCANNED = (1u << 6u),
    REACHED = (1u << 7u),
    COLLECTED = (1u << 8u)
  };

public:
  /**
   * Get the class name.
   */
  virtual bi::type::String getClassName() const {
    return "Any";
  }

  /**
   * Size of the object.
   */
  virtual unsigned size_() const {
    return sizeof(*this);
  }

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
  virtual void recycle_(Label* label) = 0;

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
   * Accept a visitor across member variables.
   */
  template<class Visitor>
  void accept_(const Visitor& v) {
    //
  }

  /**
   * Type of members.
   */
  using member_type_ = decltype(label);
};
}
