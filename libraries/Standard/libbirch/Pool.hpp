/**
 * @file
 */
#pragma once

#include "libbirch/Lock.hpp"
#include "libbirch/memory.hpp"

namespace libbirch {
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
  Pool() :
      top(nullptr) {
    //
  }

  /**
   * Is the pool empty?
   */
  bool empty() const {
    return !top;
  }

  /**
   * Pop an allocation from the pool. Returns `nullptr` if the pool is
   * empty.
   */
  void* pop() {
    lock.set();
    auto result = top;
    top = getNext(result);
    lock.unset();
    return result;
  }

  /**
   * Push an allocation to the pool.
   */
  void push(void* block) {
    lock.set();
    setNext(block, top);
    top = block;
    lock.unset();
  }

private:
  /**
   * Get the first 8 bytes of a block as a pointer.
   */
  static void* getNext(void* block) {
    return (block) ? *reinterpret_cast<void**>(block) : nullptr;
  }

  /**
   * Set the first 8 bytes of a block as a pointer.
   */
  static void setNext(void* block, void* next) {
    assert(block);
    *reinterpret_cast<void**>(block) = next;
  }

  /**
   * Stack of allocations.
   */
  void* top;

  /**
   * Mutex.
   */
  Lock lock;
};
}
