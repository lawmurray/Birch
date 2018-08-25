/**
 * @file
 */
#pragma once

#include <atomic>

namespace bi {
/**
 * Lock with shared and exclusive use semantics.
 *
 * @ingroup libbirch
 */
class Lock {
public:
  /**
   * Constructor.
   */
  Lock();

  /**
   * Obtain shared use.
   */
  void share();

  /**
   * Release shared use.
   */
  void unshare();

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
   * Lock type.
   */
  struct lock_type {
    /**
     * Count of threads with shared access.
     */
    unsigned shareCount;

    /**
     * Count of threads with exclusive access.
     */
    unsigned keepCount;
  };

  /**
   * Lock.
   */
  std::atomic<lock_type> lock;
};
}
