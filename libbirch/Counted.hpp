/**
 * @file
 */
#pragma once

#include "libbirch/global.hpp"

namespace bi {
/**
 * Base class for reference counted objects.
 *
 * @ingroup libbirch
 */
class Counted {
public:
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
   * Destroy the object. This is virtual in order that it is called on
   * the object of the most-derived type. That will set ptr and size
   * to the correct values for later use in deallocate().
   */
  virtual void destroy();

  /**
   * Deallocate the object.
   */
  void deallocate();

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
   * Increment the weak count.
   */
  void incWeak();

  /**
   * Decrement the weak count.
   */
  void decWeak();

protected:
  /**
   * Pointer to allocation for this object.
   */
  void* ptr;

  /**
   * Size of the object.
   */
  size_t size;

  /**
   * Shared count.
   */
  unsigned sharedCount;

  /**
   * Weak count.
   */
  unsigned weakCount;
};
}

#include "libbirch/global.hpp"

inline bi::Counted::Counted() :
    ptr(nullptr),
    size(0),
    sharedCount(0),
    weakCount(1) {
  //
}

inline bi::Counted::Counted(const Counted& o) :
    ptr(nullptr),
    size(0),
    sharedCount(0),
    weakCount(1) {
  //
}

inline bi::Counted::~Counted() {
  assert(sharedCount == 0);
}

inline void bi::Counted::destroy() {
  this->ptr = this;
  this->size = sizeof(*this);
  this->~Counted();
}

inline void bi::Counted::deallocate() {
  bi::deallocate(ptr, size);
}

inline bi::Counted* bi::Counted::lock() {
  unsigned count;
  #pragma omp atomic capture
  {
    sharedCount += sharedCount > 0 ? 1 : 0;
    count = sharedCount;
  }
  return count > 0 ? this : nullptr;
}

inline void bi::Counted::incShared() {
  #pragma omp atomic update
  ++sharedCount;
}

inline void bi::Counted::decShared() {
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

inline void bi::Counted::incWeak() {
  #pragma omp atomic update
  ++weakCount;
}

inline void bi::Counted::decWeak() {
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
