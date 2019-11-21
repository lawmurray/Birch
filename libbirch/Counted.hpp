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
      weakCount(1u),
      memoCount(1u) {
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
  unsigned getSize() const;

  /**
   * Used by a shared pointer when it is known that the object has not yet
   * been assigned to any smart pointer. Sets the shared count to one, but
   * not atomically.
   */
  void init();

  /**
   * Increment the shared count.
   */
  void incShared();

  /**
   * Decrement the shared count.
   */
  void decShared();

  /**
   * Increment the shared count twice, as one operation.
   */
  void doubleIncShared();

  /**
   * Decrement the shared count twice, as one operation.
   */
  void doubleDecShared();

  /**
   * Shared count.
   */
  unsigned numShared() const;

  /**
   * Increment the weak count.
   */
  void incWeak();

  /**
   * Decrement the weak count.
   */
  void decWeak();

  /**
   * Weak count.
   */
  unsigned numWeak() const;

  /**
   * Increment the memo count (implies an increment of the weak count also).
   */
  void incMemo();

  /**
   * Decrement the memo count (implies a decrement of the weak count also).
   */
  void decMemo();

  /**
   * Memo count.
   */
  unsigned numMemo() const;

  /**
   * Is this object reachable? An object is reachable if it contains a shared
   * count of one or more, or a weak count greater than the memo count. When
   * the weak count equals the memo count (it cannot be less), the object
   * is only reachable via keys in memos, which will never be triggered, and
   * so the object is not considered reachable.
   */
  bool isReachable() const;

  /**
   * Name of the class.
   */
  virtual const char* getClassName() const {
    return "Counted";
  }

protected:
  /**
   * Destroy, but do not deallocate, the object.
   */
  void destroy() {
    assert(sharedCount.load() == 0u);
    this->~Counted();
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
   * Shared count.
   */
  Atomic<unsigned> sharedCount;

  /**
   * Weak count. This is one plus the number of times that the object is held
   * by a weak pointer. The plus one is a self-reference that is released
   * when the shared count reaches zero.
   */
  Atomic<unsigned> weakCount;

  /**
   * Memo count. This is one plus the number of times that the object occurs
   * as a key in a memo. The plus one is a self-reference that is relased
   * when the weak count reaches zero.
   */
  Atomic<unsigned> memoCount;

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

#include "libbirch/thread.hpp"

inline unsigned libbirch::Counted::getSize() const {
  return size;
}

inline void libbirch::Counted::init() {
  assert(sharedCount.load() == 0u);
  sharedCount.init(1u);
}

inline void libbirch::Counted::incShared() {
  //assert(sharedCount.load() > 0u);
  sharedCount.increment();
}

inline void libbirch::Counted::decShared() {
  assert(sharedCount.load() > 0u);
  if (--sharedCount == 0u) {
    destroy();
    decWeak();  // release weak self-reference
  }
}

inline void libbirch::Counted::doubleIncShared() {
  //assert(sharedCount.load() > 0u);
  sharedCount.doubleIncrement();
}

inline void libbirch::Counted::doubleDecShared() {
  assert(sharedCount.load() > 0u);
  if ((sharedCount -= 2u) == 0u) {
    destroy();
    decWeak();  // release weak self-reference
  }
}

inline unsigned libbirch::Counted::numShared() const {
  return sharedCount.load();
}

inline void libbirch::Counted::incWeak() {
  assert(weakCount.load() > 0u);
  weakCount.increment();
}

inline void libbirch::Counted::decWeak() {
  assert(weakCount.load() > 0u);
  if (--weakCount == 0u) {
    assert(sharedCount.load() == 0u);
    decMemo();  // release memo self-reference
  }
}

inline unsigned libbirch::Counted::numWeak() const {
  return weakCount.load();
}

inline void libbirch::Counted::incMemo() {
  memoCount.increment();
}

inline void libbirch::Counted::decMemo() {
  assert(memoCount.load() > 0u);
  if (--memoCount == 0u) {
    assert(sharedCount.load() == 0u);
    assert(weakCount.load() == 0u);
    deallocate();
  }
}

inline unsigned libbirch::Counted::numMemo() const {
  return memoCount.load();
}

inline bool libbirch::Counted::isReachable() const {
  return numWeak() > 0u;
}
