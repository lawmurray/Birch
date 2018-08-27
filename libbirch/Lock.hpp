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
   * Joint lock type.
   */
  struct joint_lock_type {
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
   * Split lock type.
   */
  struct split_lock_type {
    /**
     * Count of threads with shared access.
     */
    std::atomic<unsigned> shareCount;

    /**
     * Count of threads with exclusive access.
     */
    std::atomic<unsigned> keepCount;
  };

  /**
   * Lock type.
   */
  union lock_type {
    std::atomic<joint_lock_type> joint;
    split_lock_type split;
  };

  /**
   * Lock.
   */
  lock_type lock;
};
}
