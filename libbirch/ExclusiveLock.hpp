/**
 * @file
 */
#pragma once

#include "libbirch/external.hpp"
#include "libbirch/Atomic.hpp"

namespace libbirch {
#pragma omp declare target
/**
 * Lock with only exclusive use semantics.
 *
 * @ingroup libbirch
 */
class ExclusiveLock {
public:
  /**
   * Constructor.
   */
  ExclusiveLock();

  /**
   * Copy constructor.
   */
  ExclusiveLock(const ExclusiveLock& o);

  /**
   * Obtain exclusive use.
   */
  void keep();

  /**
   * Release exclusive use.
   */
  void unkeep();

private:
  /**
   * Lock.
   */
  Atomic<bool> lock;
};
#pragma omp end declare target
}

inline libbirch::ExclusiveLock::ExclusiveLock() :
    lock(false) {
  //
}

inline libbirch::ExclusiveLock::ExclusiveLock(const ExclusiveLock& o) :
    lock(false) {
  //
}

inline void libbirch::ExclusiveLock::keep() {
  /* spin, setting the lock true until its old value comes back false */
  while (lock.exchange(true));
}

inline void libbirch::ExclusiveLock::unkeep() {
  lock.store(false);
}
