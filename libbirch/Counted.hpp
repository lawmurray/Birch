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
   * Create a shared pointer from this. If this has been destroyed
   * (but still exists due to weak pointers), returns a null pointer.
   */
  template<class T>
  SharedPtr<T> lock();

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

template<class T>
inline bi::SharedPtr<T> bi::Counted::lock() {
  if (sharedCount > 0) {
    return SharedPtr<T>(static_cast<T*>(this));
  } else {
    return SharedPtr<T>();
  }
}

inline void bi::Counted::incShared() {
  ++sharedCount;
}

inline void bi::Counted::decShared() {
  assert(sharedCount > 0);
  --sharedCount;
  if (sharedCount == 0) {
    destroy();
    decWeak();
  }
}

inline void bi::Counted::incWeak() {
  ++weakCount;
}

inline void bi::Counted::decWeak() {
  assert(weakCount > 0);
  --weakCount;
  if (weakCount == 0) {
    assert(sharedCount == 0);
    // ^ objects keep a weak pointer to themselves, so the weak count
    //   should not expire before the shared count
    deallocate();
  }
}
