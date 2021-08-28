/**
 * @file
 */
#pragma once

#include "numbirch/array/external.hpp"

namespace numbirch {
/**
 * @internal
 * 
 * Lock with exclusive use semantics.
 */
class Lock {
public:
  /**
   * Constructor.
   */
  Lock() {
    #pragma omp atomic write
    lock = false;
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
    bool old;
    do {
      #pragma omp atomic capture seq_cst
      {
        old = lock;
        lock = true;
      }
    } while (old);
  }

  /**
   * Release exclusive use.
   */
  void unset() {
    #pragma omp atomic write seq_cst
    lock = false;
  }

private:
  /**
   * Lock.
   */
  bool lock;
};
}
