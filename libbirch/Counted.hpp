/**
 * @file
 */
#pragma once

#include "libbirch/config.hpp"

namespace bi {
/**
 * Base class for reference counted objects.
 *
 * @attention In order to work correctly, Counted must be the *first* base
 * class in any inheritance hierarchy. This is particularly important when
 * multiple inheritance is used.
 *
 * @ingroup libbirch
 */
class Counted {
protected:
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
   * Assignment operator.
   */
  Counted& operator=(const Counted&) = delete;

public:
  /**
   * Create an object,
   */
  template<class... Args>
  static Counted* create(Args... args) {
    return emplace(allocate<sizeof(Counted)>(), args...);
  }

  /**
   * Create an object in previously-allocated memory.
   */
  template<class... Args>
  static Counted* emplace(void* ptr, Args... args) {
    auto o = new (ptr) Counted();
    o->size = sizeof(Counted);
    return o;
  }

  /**
   * Clone the object.
   */
  virtual Counted* clone() const {
    return emplace(allocate<sizeof(Counted)>(), *this);
  }

  /**
   * Clone the object into previous allocation.
   */
  virtual Counted* clone(void* ptr) const {
    return emplace(ptr, *this);
  }

  /**
   * Destroy the object.
   */
  virtual void destroy() {
    this->~Counted();
  }

  /**
   * Deallocate the object.
   */
  void deallocate();

  /**
   * Get the size, in bytes, of the object.
   */
  unsigned getSize() const;

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
   * Shared count.
   */
  unsigned numShared() const;

  /**
   * Increment the weak count.
   */
  void incWeak();

  /**
   * Decrement the weak count.
   */
  void decWeak();

  /**
   * Weak count.
   */
  unsigned numWeak() const;

  /**
   * Is this shared?
   */
  bool isShared() const;

protected:
  /**
   * Size of the object.
   */
  unsigned size;

  /**
   * Shared count.
   */
  std::atomic<unsigned> sharedCount;

  /**
   * Weak count.
   */
  std::atomic<unsigned> weakCount;
};
}
