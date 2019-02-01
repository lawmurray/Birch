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

bi::Counted* bi::Counted::lock() {
  unsigned count = sharedCount.load(std::memory_order_acquire);
  while (count > 0u
      && !sharedCount.compare_exchange_weak(count, count + 1u,
          std::memory_order_release)) {
    //
  }
  return count > 0u ? this : nullptr;
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

void bi::Counted::decWeak() {
  assert(weakCount > 0u);
  if (weakCount.fetch_sub(1u, std::memory_order_relaxed) - 1u == 0u) {
    assert(sharedCount == 0u);
    // ^ because of weak self-reference, the weak count should not expire
    //   before the shared count
    deallocate();
  }
}

void bi::Counted::freeze() {
  bool expected = false;
  bool desired = true;
  if (frozen.compare_exchange_strong(expected, desired,
      std::memory_order_relaxed)) {
    doFreeze();
  }
}

bool bi::Counted::isReachable() const {
  return sharedCount.load(std::memory_order_relaxed) > 0u
      || weakCount.load(std::memory_order_relaxed)
          > memoCount.load(std::memory_order_relaxed);
}
