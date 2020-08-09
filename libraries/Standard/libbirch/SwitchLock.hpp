/**
 * @file
 */
#pragma once

#include "libbirch/external.hpp"
#include "libbirch/Lock.hpp"
#include "libbirch/Semaphore.hpp"

namespace libbirch {
/**
 * Lock allowing multiple tasks. Any number of threads may perform the same
 * task concurrently, but two or more tasks cannot be performed concurrently.
 *
 * @ingroup libbirch
 */
class SwitchLock {
public:
  /**
   * Constructor.
   */
  SwitchLock() :
      nleft(0u),
      nright(0u),
      direction(NONE) {
    //
  }

  /**
   * Copy constructor.
   */
  SwitchLock(const SwitchLock&) : SwitchLock() {
    //
  }

  /**
   * Correctly initialize after a bitwise copy.
   */
  void bitwiseFix() {
    lock.bitwiseFix();
    left.bitwiseFix();
    right.bitwiseFix();
    nleft.set(0u);
    nright.set(0u);
    direction.set(NONE);
  }

  /**
   * Enter left.
   */
  void enterLeft() {
    lock.set();
    nleft.increment();
    auto d = direction.load();
    if (d == NONE) {
      d = LEFT;
      direction.store(LEFT);
    }
    if (d == LEFT) {
      left.release();  // permit entry to self, acquired below
    }
    lock.unset();
    left.acquire();
  }

  /**
   * Exit left.
   */
  void exitLeft() {
    lock.set();
    if (--nleft == 0u) {  // no remaining workers on left
      auto n = nright.load();
      if (n > 0u) {
        direction.store(RIGHT);
        right.release(n);
      } else {
        direction.store(NONE);
      }
    }
    lock.unset();
  }

  /**
   * Enter right.
   */
  void enterRight() {
    lock.set();
    nright.increment();
    auto d = direction.load();
    if (d == NONE) {
      d = RIGHT;
      direction.store(RIGHT);
    }
    if (d == RIGHT) {
      right.release();  // permit entry to self, acquired below
    }
    lock.unset();
    right.acquire();
  }

  /**
   * Exit right.
   */
  void exitRight() {
    lock.set();
    if (--nright == 0u) {  // no remaining workers on right
      auto n = nleft.load();
      if (n > 0u) {
        direction.store(LEFT);
        left.release(n);
      } else {
        direction.store(NONE);
      }
    }
    lock.unset();
  }

private:
  /**
   * Possible directions.
   */
  enum Direction {
    NONE,
    LEFT,
    RIGHT
  };

  /**
   * Lock.
   */
  Lock lock;

  /**
   * Semaphore for left traffic.
   */
  Semaphore left;

  /**
   * Semaphore for right traffic.,
   */
  Semaphore right;

  /**
   * Number of threads on left.
   */
  Atomic<unsigned> nleft;

  /**
   * Number of threads on right.
   */
  Atomic<unsigned> nright;

  /**
   * Current direction.
   */
  Atomic<Direction> direction;
};
}
