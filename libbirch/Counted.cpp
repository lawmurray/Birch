/**
 * @file
 */
#include "libbirch/Counted.hpp"

bi::Counted::Counted() :
    sharedCount(0),
    weakCount(1),
    memoCount(0),
    size(0),
    key(nullptr) {
  //
}

bi::Counted::Counted(const Counted& o) :
    sharedCount(0),
    weakCount(1),
    memoCount(0),
    size(o.size),
    key(&o) {
  //
}

bi::Counted::~Counted() {
  assert(sharedCount == 0);
  assert(memoCount == 0);
}

void bi::Counted::deallocate() {
  assert(sharedCount == 0);
  assert(weakCount == 0);
  assert(memoCount == 0);
  bi::deallocate(this, size);
}

unsigned bi::Counted::getSize() const {
  return size;
}

bi::Counted* bi::Counted::lock() {
  if (memoCount > 0) {
    ++sharedCount;
    return this;
  } else {
    unsigned count = sharedCount;
    while (count > 0 && !sharedCount.compare_exchange_weak(count, count + 1)) {
      //
    }
    return count > 0 ? this : nullptr;
  }
}

void bi::Counted::incShared() {
  ++sharedCount;
}

void bi::Counted::decShared() {
  assert(sharedCount > 0);
  if (--sharedCount == 0) {
    if (memoCount) {
      if (!key->isReachable()) {
        releaseMemo();
      }
    } else {
      destroy();
      decWeak();  // release weak self-reference
    }
  }
}

unsigned bi::Counted::numShared() const {
  return sharedCount;
}

void bi::Counted::incWeak() {
  ++weakCount;
}

void bi::Counted::decWeak() {
  assert(weakCount > 0);
  if (--weakCount == 0) {
    assert(sharedCount == 0);
    assert(memoCount == 0);
    // ^ because of weak self-reference, the weak count should not expire
    //   before the other counts
    deallocate();
  }
}

unsigned bi::Counted::numWeak() const {
  return weakCount;
}

void bi::Counted::setMemo() {
  assert(memoCount == 0);
  assert(key);
  memoCount = 1;
}

void bi::Counted::releaseMemo() {
  unsigned value = 1u;
  if (memoCount.compare_exchange_strong(value, 0u)) {
    ///@todo Race-condition with decShared()?
    if (sharedCount == 0) {
      destroy();
      decWeak();  // release weak self-reference
    }
  }
}

bool bi::Counted::hasMemo() const {
  return memoCount == 1;
}

bool bi::Counted::isReachable() const {
  return sharedCount > 0 ||
      (memoCount == 0 && weakCount > 1) ||
      // ^ not in memo, more than self reference
      (memoCount > 0 && weakCount > 2);
      // ^ in memo, more than self reference and key reference
}
