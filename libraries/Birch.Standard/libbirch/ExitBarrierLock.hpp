/**
 * @file
 */
#pragma once

#include "libbirch/Atomic.hpp"

namespace libbirch {
/**
 * Entry-exit lock.
 *
 * @ingroup libbirch
 *
 * Permits any number of threads to enter a critical region, but has an exit
 * barrier that will not allow them to exit it until all such threads have
 * reached the end of the critical region.
 *
 * This is used for thread safety in the particular case of *finish* and
 * *freeze* operations on objects.
 */
class ExitBarrierLock {
public:
  /**
   * Default constructor.
   */
  ExitBarrierLock();

  /**
   * Enter the critical region, possibly blocking.
   */
  void enter();

  /**
   * Exit the critical region, possibly blocking.
   */
  void exit();

private:
  /**
   * Number of threads in critical region.
   */
  Atomic<unsigned> ninternal;
};
}

inline libbirch::ExitBarrierLock::ExitBarrierLock() :
    ninternal(0) {
  //
}

inline void libbirch::ExitBarrierLock::enter() {
  ++ninternal;
}

inline void libbirch::ExitBarrierLock::exit() {
  if (--ninternal == 0) {
    return;
  } else {
    while (ninternal.load() != 0);  // spin until the entry gate is open
  }
}
