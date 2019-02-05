/**
 * @file
 */
#pragma once

#include "libbirch/config.hpp"

namespace bi {
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
  std::atomic<bool> lock;
};
}

inline bi::ExclusiveLock::ExclusiveLock() :
    lock(false) {
  //
}

inline bi::ExclusiveLock::ExclusiveLock(const ExclusiveLock& o) :
    lock(false) {
  //
}

inline void bi::ExclusiveLock::unkeep() {
  lock.store(false, std::memory_order_seq_cst);
}
