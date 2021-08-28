/**
 * @file
 */
#pragma once

#include "numbirch/array/external.hpp"

namespace numbirch {
/*
 * Control block for reference counting of Array buffers.
 */
class ArrayControl {
public:
  /**
   * Constructor.
   *
   * @param r Initial reference count.
   */
  ArrayControl(const int r) {
    #pragma omp atomic write
    r_ = r;
  }

  /**
   * Reference count.
   */
  int numShared_() const {
    int r;    
    #pragma omp atomic read
    r = r_;
    return r;
  }

  /**
   * Increment the shared reference count.
   */
  void incShared_() {
    assert(numShared_() > 0);
    #pragma omp atomic update
    ++r_;
  }

  /**
   * Decrement the shared reference count and return the new value.
   */
  int decShared_() {
    int r;
    #pragma omp atomic capture
    {
      --r_;
      r = r_;
    }
    return r;
  }

private:
  /**
   * Reference count.
   */
  int r_;
};
}
