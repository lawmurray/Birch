/**
 * @file
 */
#if ENABLE_MEMORY_POOL
#pragma once

#include "libbirch/ExclusiveLock.hpp"

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

/**
 * Get the @i th pool.
 */
extern Pool& pool(const unsigned i);

/**
 * Buffer for heap allocations.
 */
extern Atomic<char*> buffer;

/**
 * Start of heap (for debugging purposes).
 */
extern char* bufferStart;

/**
 * Size of heap (for debugging purposes).
 */
extern size_t bufferSize;

}

inline libbirch::Pool::Pool() :
    top(nullptr) {
  //
}

inline bool libbirch::Pool::empty() const {
  return !top;
}

inline void* libbirch::Pool::pop() {
  lock.keep();
  auto result = top;
  top = getNext(result);
  lock.unkeep();
  return result;
}

inline void libbirch::Pool::push(void* block) {
  assert(bufferStart <= block && block < bufferStart + bufferSize);
  lock.keep();
  setNext(block, top);
  top = block;
  lock.unkeep();
}

inline void* libbirch::Pool::getNext(void* block) {
  assert(
      !block || (bufferStart <= block && block < bufferStart + bufferSize));

  return (block) ? *reinterpret_cast<void**>(block) : nullptr;
}

inline void libbirch::Pool::setNext(void* block, void* value) {
  assert(block);
  assert(bufferStart <= block && block < bufferStart + bufferSize);
  assert(
      !value || (bufferStart <= value && value < bufferStart + bufferSize));

  *reinterpret_cast<void**>(block) = value;
}

#endif
