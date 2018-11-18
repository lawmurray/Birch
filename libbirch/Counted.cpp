/**
 * @file
 */
#include "libbirch/Counted.hpp"

bi::Counted::Counted() :
    size(0),
    sharedCount(0),
    weakCount(1) {
  //
}

bi::Counted::Counted(const Counted& o) :
    size(0),
    sharedCount(0),
    weakCount(1) {
  //
}

bi::Counted::~Counted() {
  assert(sharedCount == 0);
}

void bi::Counted::deallocate() {
  bi::deallocate(this, size);
}

bi::Counted* bi::Counted::lock() {
  unsigned count = sharedCount;
  while (count > 0 && !sharedCount.compare_exchange_weak(count, count + 1)) {
    //
  }
  return count > 0 ? this : nullptr;
}

void bi::Counted::incShared() {
  sharedCount.fetch_add(1u);
}

void bi::Counted::decShared() {
  assert(sharedCount > 0);
  if (sharedCount.fetch_sub(1u) == 1) {
    destroy();
    decWeak();
  }
}

unsigned bi::Counted::numShared() const {
  return sharedCount.load();
}

void bi::Counted::incWeak() {
  weakCount.fetch_add(1u);
}

void bi::Counted::decWeak() {
  assert(weakCount > 0);
  if (weakCount.fetch_sub(1u) == 1) {
    assert(sharedCount == 0);
    // ^ objects keep a weak pointer to themselves, so the weak count
    //   should not expire before the shared count
    deallocate();
  }
}

unsigned bi::Counted::numWeak() const {
  return weakCount.load();
}

bool bi::Counted::isShared() const {
  return sharedCount > 1;
}
