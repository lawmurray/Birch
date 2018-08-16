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
  unsigned count;
  #pragma omp atomic capture
  {
    sharedCount += sharedCount > 0 ? 1 : 0;
    count = sharedCount;
  }
  return count > 0 ? this : nullptr;
}

void bi::Counted::incShared() {
  #pragma omp atomic update
  ++sharedCount;
}

void bi::Counted::decShared() {
  assert(sharedCount > 0);

  unsigned count;
  #pragma omp atomic capture
  {
    --sharedCount;
    count = sharedCount;
  }
  if (count == 0) {
    destroy();
    decWeak();
  }
}

void bi::Counted::incWeak() {
  #pragma omp atomic update
  ++weakCount;
}

void bi::Counted::decWeak() {
  assert(weakCount > 0);

  unsigned count;
  #pragma omp atomic capture
  {
    --weakCount;
    count = weakCount;
  }
  if (count == 0) {
    assert(sharedCount == 0);
    // ^ objects keep a weak pointer to themselves, so the weak count
    //   should not expire before the shared count
    deallocate();
  }
}

bool bi::Counted::isShared() const {
  return sharedCount > 1;
}
