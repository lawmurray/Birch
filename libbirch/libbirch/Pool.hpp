/**
 * @file
 */
#pragma once

#include "libbirch/Lock.hpp"
#include "libbirch/memory.hpp"

namespace libbirch {
/**
 * Pool of memory blocks.
 *
 * @ingroup libbirch
 * 
 * The pool is intended to manage allocations of the same size. It maintains
 * a free list of memory blocks returned to the pool that it may later reuse.
 * The pool is considered to be owned by a particular thread, managed
 * externally. This owning thread takes blocks from the pool, and returns them
 * to the pool, lock free. Other threads cannot take blocks from the pool, but
 * may return them; a lock is required internally when this occurs.
 */
class Pool {
public:
  /**
   * Constructor.
   */
  Pool() :
      freeList(nullptr),
      pendList(nullptr) {
    //
  }

  /**
   * Take an allocation from the pool. Returns `nullptr` if the pool is
   * empty.
   */
  void* take() {
    if (!freeList) {
      /* free list is empty, swap in the pend list instead */
      lock.set();
      freeList = pendList;
      pendList = nullptr;
      lock.unset();
    }
    void* block = freeList;
    freeList = getNext(block);
    return block;
  }

  /**
   * Return an allocation to the free list. This is used by the thread that
   * owns the pool.
   */
  void free(void* block) {
    setNext(block, freeList);
    freeList = block;
  }

  /**
   * Return an allocation to the pend list. This is used by a thread that does
   * not own the pool.
   */
  void pend(void* block) {
    lock.set();
    setNext(block, pendList);
    pendList = block;
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
   * Free list.
   */
  void* freeList;

  /**
   * Pend list. Blocks returned to the pool by threads that do not own the
   * pool are enqueued here first, and later returned to the free list by the
   * owning thread.
   */
  void* pendList;

  /**
   * Mutex.
   */
  Lock lock;
};
}
