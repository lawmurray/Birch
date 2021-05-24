/**
 * @file
 */
#pragma once

#include "libbirch/external.hpp"
#include "libbirch/Atomic.hpp"

namespace libbirch {
/**
 * Control block for reference counting of Array buffers.
 *
 * @ingroup libbirch
 */
class ArrayControl {
public:
  /**
   * Constructor.
   *
   * @param r Initial reference count.
   */
  ArrayControl(const int r) : r_(r) {
    //
  }

  /**
   * New operator.
   */
  void* operator new(std::size_t size) {
    /* object destruction and deallocation are separated; an explicit call to
     * the destructor is used to destroy, and std::free() used to deallocate,
     * so std::malloc() should be used to allocate; using the default operator
     * new can result in a double free, as reported by valgrind */
    return std::malloc(size);
  }

  /**
   * Delete operator.
   */
  void operator delete(void* ptr) {
    std::free(ptr);
  }

  /**
   * Reference count.
   */
  int numShared_() const {
    return r_.load();
  }

  /**
   * Increment the shared reference count.
   */
  void incShared_() {
    assert(numShared_() > 0);
    r_.increment();
  }

  /**
   * Decrement the shared reference count and return the new value.
   */
  int decShared_() {
    return --r_;
  }

private:
  /**
   * Reference count.
   */
  Atomic<int> r_;
};
}
