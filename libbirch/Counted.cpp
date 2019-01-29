/**
 * @file
 */
#include "libbirch/Counted.hpp"

bi::Counted::Counted() :
    sharedCount(0u),
    weakCount(1u),
    memoCount(0u),
    size(0u),
    frozen(false) {
  //
}

bi::Counted::Counted(const Counted& o) :
    sharedCount(0u),
    weakCount(1u),
    memoCount(0u),
    size(o.size),
    frozen(false) {
  //
}

bi::Counted::~Counted() {
  assert(sharedCount == 0u);
}

void bi::Counted::deallocate() {
  assert(sharedCount == 0u);
  assert(weakCount == 0u);
  bi::deallocate(this, size);
}

unsigned bi::Counted::getSize() const {
  return size;
}

bi::Counted* bi::Counted::lock() {
  unsigned count = sharedCount.load(std::memory_order_acquire);
  while (count > 0u
      && !sharedCount.compare_exchange_weak(count, count + 1u,
          std::memory_order_release)) {
    //
  }
  return count > 0u ? this : nullptr;
}

void bi::Counted::incShared() {
  sharedCount.fetch_add(1u, std::memory_order_relaxed);
}

void bi::Counted::decShared() {
  assert(sharedCount > 0u);
  if (sharedCount.fetch_sub(1u, std::memory_order_relaxed) - 1u == 0u
      && size > 0u) {
    // ^ size == 0u during construction, never destroy in that case
    destroy();
    decWeak();  // release weak self-reference
  }
}

unsigned bi::Counted::numShared() const {
  return sharedCount;
}

void bi::Counted::incWeak() {
  weakCount.fetch_add(1u, std::memory_order_relaxed);
}

void bi::Counted::decWeak() {
  assert(weakCount > 0u);
  if (weakCount.fetch_sub(1u, std::memory_order_relaxed) - 1u == 0u) {
    assert(sharedCount == 0u);
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
  memoCount.fetch_add(1u, std::memory_order_relaxed);
}

void bi::Counted::decMemo() {
  /* the order of operations here is important, as the weak count should
   * never be less than the memo count */
  assert(memoCount > 0u);
  memoCount.fetch_sub(1u, std::memory_order_relaxed);
  decWeak();
}

bool bi::Counted::isReachable() const {
  return sharedCount.load(std::memory_order_relaxed) > 0u
      || weakCount.load(std::memory_order_relaxed)
          > memoCount.load(std::memory_order_relaxed);
}

bool bi::Counted::isFrozen() const {
  return frozen.load(std::memory_order_relaxed);
}

void bi::Counted::freeze() {
  bool expected = false;
  bool desired = true;
  if (frozen.compare_exchange_strong(expected, desired,
      std::memory_order_relaxed)) {
    doFreeze();
  }
}

void bi::Counted::doFreeze() {
  //
}
