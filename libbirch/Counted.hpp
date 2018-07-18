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
   * Deallocate the memory for the object.
   */
  virtual void deallocate();

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
   * Shared count.
   */
  unsigned sharedCount;

  /**
   * Weak count.
   */
  unsigned weakCount;
};
}

#include "libbirch/PowerPoolAllocator.hpp"
#include "libbirch/global.hpp"

inline bi::Counted::Counted() :
    sharedCount(0),
    weakCount(1) {
  //
}

inline bi::Counted::Counted(const Counted& o) :
    sharedCount(0),
    weakCount(1) {
  //
}

inline bi::Counted::~Counted() {
  assert(sharedCount == 0);
  decWeak();
}

inline void bi::Counted::deallocate() {
  bi::deallocate(this, sizeof(this));
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
  --sharedCount;
  if (sharedCount == 0) {
    this->~Counted();
  }
}

inline void bi::Counted::incWeak() {
  ++weakCount;
}

inline void bi::Counted::decWeak() {
  --weakCount;
  if (weakCount == 0) {
    assert(sharedCount == 0);
    // ^ objects keep a weak pointer to themselves, so the weak count
    //   should not expire before the shared count
    deallocate();
  }
}
