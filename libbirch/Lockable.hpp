/**
 * @file
 */
#pragma once

#ifdef _OPENMP
#include <omp.h>
#endif

namespace bi {
/**
 * Lockable object.
 *
 * @ingroup libbirch
 */
class Lockable {
public:
  /**
   * Constructor.
   */
  Lockable() {
    #ifdef _OPENMP
    omp_init_lock(&mutex);
    #endif
  }

  /**
   * Destructor.
   */
  ~Lockable() {
    #ifdef _OPENMP
    omp_destroy_lock(&mutex);
    #endif
  }

  /**
   * Set the lock.
   */
  void set() {
    #ifdef _OPENMP
    omp_set_lock(&mutex);
    #endif
  }

  /**
   * Unset the lock.
   */
  void unset() {
    #ifdef _OPENMP
    omp_unset_lock(&mutex);
    #endif
  }

private:
  #ifdef _OPENMP
  /**
   * Lock.
   */
  omp_lock_t mutex;
  #endif
};
}
