/**
 * @file
 */
#pragma once

#include "libbirch/external.hpp"

namespace libbirch {
/**
 * Lock with shared and exclusive use semantics.
 *
 * @ingroup libbirch
 *
 * @todo Could replace with std::shared_mutex in C++17.
 */
class Lock {
public:
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
    Atomic<unsigned> shareCount;

    /**
     * Count of threads with exclusive access.
     */
    Atomic<unsigned> keepCount;
  };

  /**
   * Lock type.
   */
  union lock_type {
    Atomic<joint_lock_type> joint;
    split_lock_type split;

    lock_type() :
        joint( { 0u, 0u }) {
      //
    }
  };

  /**
   * Lock.
   */
  lock_type lock;
};
}

inline void libbirch::Lock::unshare() {
  lock.split.shareCount.fetch_sub(1u, std::memory_order_seq_cst);
}

inline void libbirch::Lock::unkeep() {
  lock.split.keepCount.store(0u, std::memory_order_seq_cst);
}
