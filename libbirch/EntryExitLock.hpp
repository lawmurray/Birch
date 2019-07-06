/**
 * @file
 */
#pragma once

#include "libbirch/external.hpp"

namespace libbirch {
/**
 * Lock that permits any number of threads to enter a critical region, but
 * as soon as any thread reaches the end of the region, no more threads are
 * allowed to enter until all threads inside have exited.
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
  Atomic<unsigned> internal;

  /**
   * Is the entrance gate open?
   */
  Atomic<bool> entry;
};
}

inline libbirch::EntryExitLock::EntryExitLock() :
    internal(0),
    entry(true) {
  //
}

inline void libbirch::EntryExitLock::enter() {
  while (!entry.load());  // spin until the entry gate is open
  ++internal;
  /* it is possible that between the spin lock and incrementing internal, a
   * whole bunch of other threads enter and exit the critical region together
   * without this thread registering; that is fine for the use cases here, as
   * long as this thread hasn't started any work before incrementing
   * internal */
}

inline void libbirch::EntryExitLock::exit() {
  entry.store(false);
  if (--internal == 0) {
    entry.store(true);
  } else {
    while (!entry.load());  // spin until the entry gate is open
  }
  /* once the gate is open, it is possible that another thread enters and
   * closes the gate again before a thread in the previous generation has
   * exited; again this isn't such a problem, it will exit eventually */
}
