/**
 * @file
 */
#pragma once

#include <atomic>

namespace libbirch {
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

inline libbirch::ExclusiveLock::ExclusiveLock() :
    lock(false) {
  //
}

inline libbirch::ExclusiveLock::ExclusiveLock(const ExclusiveLock& o) :
    lock(false) {
  //
}

inline void libbirch::ExclusiveLock::keep() {
  /* spin until exclusive lock obtained */
  bool expected;
  do {
    expected = false;
  } while (!lock.compare_exchange_weak(expected, true));
}

inline void libbirch::ExclusiveLock::unkeep() {
  lock.store(false, std::memory_order_seq_cst);
}
