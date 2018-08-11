/**
 * @file
 */
#pragma once

#ifdef HAVE_OMP_H
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
    #ifdef HAVE_OMP_H
    omp_init_lock(&mutex);
    #endif
  }

  /**
   * Destructor.
   */
  ~Lockable() {
    #ifdef HAVE_OMP_H
    omp_destroy_lock(&mutex);
    #endif
  }

  /**
   * Set the lock.
   */
  void set() {
    #ifdef HAVE_OMP_H
    omp_set_lock(&mutex);
    #endif
  }

  /**
   * Unset the lock.
   */
  void unset() {
    #ifdef HAVE_OMP_H
    omp_unset_lock(&mutex);
    #endif
  }

private:
  #ifdef HAVE_OMP_H
  /**
   * Lock.
   */
  omp_lock_t mutex;
  #endif
};
}
