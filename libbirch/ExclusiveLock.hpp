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

inline void bi::ExclusiveLock::unkeep() {
  lock.store(false, std::memory_order_relaxed);
}
