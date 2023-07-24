/**
 * @file
 */
#pragma once

#include "numbirch/memory.hpp"

namespace numbirch {
/**
 * Scheduling mutex ownership wrapper.
 * 
 * @ingroup array
 *
 * For asynchronous multistreaming backends, holds an exclusive lock on the
 * scheduling mutex if any array buffers are to be moved between streams, and
 * a shared lock otherwise. Typically used before enqueuing kernels with array
 * arguments that are potentially shared between multiple streams (such
 * as arguments to a function) to ensure transactional updates of owning
 * streams, i.e. obtain the lock, update streams of arrays to the stream of
 * the current thread, enqueue kernel, release lock. See e.g. CUDA backend
 * implementation.
 * 
 * Without scheduling locks, it would be possible for the updates of owning
 * streams and the enqueuing of kernels to interleave in incorrect ways in a
 * multithreading environment.
 */
class Lock {
public:
  /**
   * Constructor.
   * 
   * @tparam ...Args Array types.
   * 
   * @param args... Arrays.
   * 
   * If the arguments are on the stream of the current thread, then obtains
   * shared ownership of the scheduling lock; otherwise, because one or more
   * arguments must be moved between streams, obtains exclusive ownership.
   * 
   * Once the lock is obtained, joins the stream of the current threads to the
   * streams of the arguments, then updates the stream of each argument to the
   * stream of the current thread.
   */
  template<typename... Args>
  Lock(Args&&... args) : exclusive(false) {
    #ifdef BACKEND_CUDA
    ((exclusive |= stream(args) && stream(args) != stream_get()), ...);
    if (exclusive) {
      lock();
      ((stream_join(args), ...));
      ((stream(args) = stream_get()), ...);
    } else {
      lock_shared();
    }
    #endif
  }

  /**
   * Destructor. Releases the lock on the scheduling mutex obtained in the
   * constructor.
   */
  ~Lock() {
    #ifdef BACKEND_CUDA
    if (exclusive) {
      unlock();
    } else {
      unlock_shared();
    }
    #endif
  }

private:
  /**
   * Is the lock exclusive?
   */
  bool exclusive;
};
}
