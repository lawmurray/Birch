/**
 * @file
 */
#pragma once

#include "libbirch/external.hpp"
#include "libbirch/Lock.hpp"

namespace libbirch {
/**
 * Semaphore.
 *
 * @ingroup libbirch
 */
class Semaphore {
public:
  /**
   * Default constructor.
   */
  Semaphore() : count(0u) {
    //
  }

  /**
   * Copy constructor.
   */
  Semaphore(const Semaphore& o) : Semaphore() {
    //
  }

  Semaphore& operator=(const Semaphore&) = delete;

  /**
   * Correctly initialize after a bitwise copy.
   */
  void bitwiseFix() {
    lock.bitwiseFix();
    count.set(0u);
  }

  /**
   * Acquire. Blocks until the count is positive, then decrements it by one.
   */
  void acquire() {
    lock.set();
    while (count.load() == 0u);
    --count;
    lock.unset();
  }

  /**
   * Release. Increments the count by one.
   */
  void release() {
    count.increment();
  }

  /**
   * Release multiple times. Increments the count by `n`.
   */
  void release(unsigned n) {
    if (n > 0) {
      count.add(n);
    }
  }

private:
  /**
   * Lock.
   */
  Lock lock;

  /**
   * Count.
   */
  Atomic<unsigned> count;
};
}
