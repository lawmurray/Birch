/**
 * @file
 */
#include "libbirch/Counted.hpp"

bi::Counted::Counted() :
    sharedCount(0),
    weakCount(1),
    memoCount(0),
    freezeCount(0),
    size(0) {
  //
}

bi::Counted::Counted(const Counted& o) :
    sharedCount(0),
    weakCount(1),
    memoCount(0),
    freezeCount(0),
    size(o.size) {
  //
}

bi::Counted::~Counted() {
  assert(sharedCount == 0);
}

void bi::Counted::deallocate() {
  assert(sharedCount == 0);
  assert(weakCount == 0);
  bi::deallocate(this, size);
}

unsigned bi::Counted::getSize() const {
  return size;
}

bi::Counted* bi::Counted::lock() {
  unsigned count = sharedCount;
  while (count > 0 && !sharedCount.compare_exchange_weak(count, count + 1)) {
    //
  }
  return count > 0 ? this : nullptr;
}

void bi::Counted::incShared() {
  sharedCount.fetch_add(1, std::memory_order_relaxed);
}

void bi::Counted::decShared() {
  assert(sharedCount > 0);
  if (--sharedCount == 0 && size > 0) {
    // ^ size == 0 during construction, never destroy in that case
    destroy();
    decWeak();  // release weak self-reference
  }
}

unsigned bi::Counted::numShared() const {
  return sharedCount;
}

void bi::Counted::incWeak() {
  weakCount.fetch_add(1, std::memory_order_relaxed);
}

void bi::Counted::decWeak() {
  assert(weakCount > 0);
  if (--weakCount == 0) {
    assert(sharedCount == 0);
    // ^ because of weak self-reference, the weak count should not expire
    //   before the shared count
    deallocate();
  }
}

unsigned bi::Counted::numWeak() const {
  return weakCount;
}

void bi::Counted::incMemo() {
  /* the order of operations here is important, as the weak count should
   * never be less than the memo count */
  incWeak();
  memoCount.fetch_add(1, std::memory_order_relaxed);
}

void bi::Counted::decMemo() {
  /* the order of operations here is important, as the weak count should
   * never be less than the memo count */
  assert(memoCount > 0);
  --memoCount;
  decWeak();
}

bool bi::Counted::isReachable() const {
  return sharedCount > 0 || weakCount > memoCount;
}

bool bi::Counted::isFrozen() const {
  unsigned n;
  do {
    n = freezeCount.load();
  } while (n > 0u && n < nthreads + 1u);
  return freezeCount > 0u;
}

void bi::Counted::freeze() {
  unsigned expected = 0u;
  unsigned desired = tid + 1;
  if (freezeCount.compare_exchange_strong(expected, desired)) {
    /* this thread freezes the object */
    doFreeze();
    freezeCount = nthreads + 1u;
  } else if (expected < nthreads + 1u) {
    /* this thread is currently freezing the object, nothing to do */
  } else if (expected < nthreads + 1u) {
    /* another thread is currently freezing the object, join it, if for no
     * other reason than to avoid a potential deadlock situation */
    doFreeze();
    freezeCount = nthreads + 1u;
  } else {
    /* already frozen */
    assert(expected == nthreads + 1u);
  }
}

void bi::Counted::doFreeze() {
  //
}
