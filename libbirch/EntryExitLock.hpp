/**
 * @file
 */
#pragma once

#include "libbirch/external.hpp"

namespace libbirch {
/**
 * Lock that permits any number of threads to enter a critical region, but
 * only exit when there are no threads in the critical region.
 *
 * @ingroup libbirch
 *
 * This is used for the peculiar case of worksharing freeze() and finish()
 * operations, ensuring that no thread returns from these until the subgraph
 * it is tasked with is definitely frozen or finished, even if some of the
 * work may have been performed by other threads with overlapping tasks.
 */
class EntryExitLock {
public:
  /**
   * Default constructor.
   */
  EntryExitLock();

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

inline libbirch::EntryExitLock::EntryExitLock() :
    ninternal(0) {
  //
}

inline void libbirch::EntryExitLock::enter() {
  ++ninternal;
}

inline void libbirch::EntryExitLock::exit() {
  if (--ninternal == 0) {
    return;
  } else {
    while (ninternal.load() != 0);  // spin until the entry gate is open
  }
}
