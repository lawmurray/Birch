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
