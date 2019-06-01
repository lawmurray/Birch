/**
 * @file
 */
#pragma once

#include "libbirch/memory.hpp"
#include "libbirch/class.hpp"

#include <atomic>

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
protected:
  /**
   * Constructor.
   */
  Counted();

  /**
   * Copy constructor.
   */
  Counted(const Counted& o);

  /**
   * Destructor.
   */
  virtual ~Counted();

  /**
   * Assignment operator.
   */
  Counted& operator=(const Counted&) = delete;

public:
  /**
   * Create an object,
   */
  template<class... Args>
  static Counted* create_(Args... args) {
    return emplace_(allocate<sizeof(Counted)>(), args...);
  }

  /**
   * Create an object in previously-allocated memory.
   */
  template<class... Args>
  static Counted* emplace_(void* ptr, Args... args) {
    auto o = new (ptr) Counted();
    o->size = sizeof(Counted);
    return o;
  }

  /**
   * Clone the object.
   */
  virtual Counted* clone_() const {
    return emplace_(allocate<sizeof(Counted)>(), *this);
  }

  /**
   * Clone the object into previous allocation.
   */
  virtual Counted* clone_(void* ptr) const {
    return emplace_(ptr, *this);
  }

  /**
   * Destroy the object.
   */
  virtual void destroy_() {
    this->~Counted();
  }

  /**
   * Deallocate the object.
   */
  void deallocate();

  /**
   * Get the size, in bytes, of the object.
   */
  unsigned getSize() const;

  /**
   * If the object has yet to be destroyed, increment the shared count
   * and return a pointer to this. Otherwise return null. This is used
   * to atomically convert a WeakPtr into a SharedPtr.
   */
  Counted* lock();

  /**
   * Increment the shared count.
   */
  void incShared();

  /**
   * Decrement the shared count.
   */
  void decShared();

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
   * Freeze this object.
   */
  void freeze();

  /**
   * Is the object reachable? An object is reachable if it contains a shared
   * count of one or more, or a weak count greater than the memo count. When
   * the weak count equals the memo count (it cannot be less), the object
   * is only reachable via keys in memos, which will never be triggered, and
   * so the object is not considered reachable.
   */
  bool isReachable() const;

  /**
   * Is the object frozen? This returns true if either a freeze is in
   * progress (i.e. another thread is in the process of freezing the object),
   * or if the freeze is complete.
   */
  bool isFrozen() const;

  /**
   * Is the object frozen, and reachable through only a single pointer?
   */
  bool isUniquelyReachable() const;

  /**
   * Name of the class.
   */
  virtual const char* name_() const {
    return "Counted";
  }

protected:
  /**
   * Perform the actual freeze of the object. This is overwritten by derived
   * classes. The non-virtual freeze() handles thread safety so that this
   * need not.
   */
  virtual void doFreeze_();

  /**
   * Shared count.
   */
  std::atomic<unsigned> sharedCount;

  /**
   * Weak count.
   */
  std::atomic<unsigned> weakCount;

  /**
   * Memo count. This is the number of times that the object occurs as a key
   * in a memo. It is always less than or equal to the weak count, as each
   * memo reference implies a weak reference also.
   */
  std::atomic<unsigned> memoCount;

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
  unsigned tid;

  /**
   * Is the object read-only? This is -1 for false, thread id for in
   * progress, max threads for true, and greater than max threads for true
   * and accessible through a single pointer only.
   */
  std::atomic<int> frozen;
};
}

#include "libbirch/thread.hpp"

inline libbirch::Counted::Counted() :
    sharedCount(0u),
    weakCount(1u),
    memoCount(0u),
    size(0u),
    tid(libbirch::tid),
    frozen(-1) {
  //
}

inline libbirch::Counted::Counted(const Counted& o) :
    sharedCount(0u),
    weakCount(1u),
    memoCount(0u),
    size(o.size),
    tid(libbirch::tid),
    frozen(-1) {
  //
}

inline libbirch::Counted::~Counted() {
  assert(sharedCount == 0u);
}

inline void libbirch::Counted::deallocate() {
  assert(sharedCount == 0u);
  assert(weakCount == 0u);
  libbirch::deallocate(this, size, tid);
}

inline unsigned libbirch::Counted::getSize() const {
  return size;
}

inline libbirch::Counted* libbirch::Counted::lock() {
  unsigned count = sharedCount.load();
  while (count > 0u
      && !sharedCount.compare_exchange_weak(count, count + 1u)) {
    //
  }
  return count > 0u ? this : nullptr;
}

inline void libbirch::Counted::incShared() {
  sharedCount.fetch_add(1u);
}

inline void libbirch::Counted::decShared() {
  assert(sharedCount > 0u);
  if (sharedCount.fetch_sub(1u) - 1u == 0u && size > 0u) {
    // ^ size == 0u during construction, never destroy in that case
    destroy_();
    decWeak();  // release weak self-reference
  }
}

inline unsigned libbirch::Counted::numShared() const {
  return sharedCount.load();
}

inline void libbirch::Counted::incWeak() {
  weakCount.fetch_add(1u);
}

inline void libbirch::Counted::decWeak() {
  assert(weakCount > 0u);
  if (weakCount.fetch_sub(1u) - 1u == 0u) {
    assert(sharedCount == 0u);
    // ^ because of weak self-reference, the weak count should not expire
    //   before the shared count
    deallocate();
  }
}

inline unsigned libbirch::Counted::numWeak() const {
  return weakCount.load();
}

inline void libbirch::Counted::incMemo() {
  /* the order of operations here is important, as the weak count should
   * never be less than the memo count */
  incWeak();
  memoCount.fetch_add(1u);
}

inline void libbirch::Counted::decMemo() {
  /* the order of operations here is important, as the weak count should
   * never be less than the memo count */
  assert(memoCount > 0u);
  memoCount.fetch_sub(1u);
  decWeak();
}

inline unsigned libbirch::Counted::numMemo() const {
  return memoCount.load();
}

inline bool libbirch::Counted::isReachable() const {
  return numWeak() > numMemo();
}

inline bool libbirch::Counted::isFrozen() const {
  return frozen.load() >= 0;
}

inline bool libbirch::Counted::isUniquelyReachable() const {
  return frozen.load() > (int)libbirch::nthreads;
}

inline void libbirch::Counted::doFreeze_() {
  //
}
