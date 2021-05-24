/**
 * @file
 */
#pragma once

#include "libbirch/external.hpp"
#include "libbirch/Atomic.hpp"

namespace libbirch {
/**
 * Lock with exclusive use semantics.
 *
 * @ingroup libbirch
 */
class Lock {
public:
  /**
   * Constructor.
   */
  Lock() :
    lock(false) {
    //
  }

  /**
   * Copy constructor.
   */
  Lock(const Lock&) : Lock() {
    //
  }

  Lock& operator=(const Lock&) = delete;

  /**
   * Obtain exclusive use.
   */
  void set() {
    /* spin, setting the lock true until its old value comes back false */
    while (lock.exchange(true));
  }

  /**
   * Release exclusive use.
   */
  void unset() {
    lock.store(false);
  }

private:
  /**
   * Lock.
   */
  Atomic<bool> lock;
};
}
