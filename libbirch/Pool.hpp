/**
 * @file
 */
#pragma once

#include "libbirch/ExclusiveLock.hpp"

namespace bi {
/**
 * Thread-safe stack of memory allocations.
 *
 * @ingroup libbirch
 *
 * The pool is kept as a stack, with blocks removed from the pool by popping
 * the stack, and returned to the pool by pushing the stack. As each
 * block is at least 8 bytes in size, when in the pool (and therefore
 * not in use), its first 8 bytes are used to store a pointer to the next
 * block on the stack. The implementation is lock-free.
 */
class Pool {
public:
  /**
   * Constructor.
   */
  Pool();

  /**
   * Is the pool empty?
   */
  bool empty() const;

  /**
   * Pop an allocation from the pool. Returns `nullptr` if the pool is
   * empty.
   */
  void* pop();

  /**
   * Push an allocation to the pool.
   */
  void push(void* block);

private:
  /**
   * Stack of allocations.
   */
  void* top;

  /**
   * Get the first 8 bytes of a block as a pointer.
   */
  static void* getNext(void* block);

  /**
   * Set the first 8 bytes of a block as a pointer.
   */
  static void setNext(void* block, void* next);

  /**
   * Mutex.
   */
  ExclusiveLock lock;
};
}
