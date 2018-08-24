/**
 * @file
 */
#include "libbirch/Counted.hpp"

#include "libbirch/global.hpp"

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

void bi::Counted::destroy() {
  this->size = sizeof(*this);
  this->~Counted();
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
  ++sharedCount;
}

void bi::Counted::decShared() {
  assert(sharedCount > 0);

  if (--sharedCount == 0) {
    destroy();
    decWeak();
  }
}

void bi::Counted::incWeak() {
  ++weakCount;
}

void bi::Counted::decWeak() {
  assert(weakCount > 0);

  if (--weakCount == 0) {
    assert(sharedCount == 0);
    // ^ objects keep a weak pointer to themselves, so the weak count
    //   should not expire before the shared count
    deallocate();
  }
}

bool bi::Counted::isShared() const {
  return sharedCount > 1;
}
