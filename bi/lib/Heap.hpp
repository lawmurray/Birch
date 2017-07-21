/**
 * @file
 */
#pragma once

#include "bi/lib/Object.hpp"

#include <vector>
#include <gc/gc_allocator.h>

namespace bi {
/**
 * Heap-local heap.
 *
 * @ingroup library
 */
class Heap {
public:
  /**
   * Destructor.
   */
  virtual ~Heap() {
    /* update fiber usage counts */
    for (auto o : heap) {
      o->disuse();
    }
  }

  /**
   * Copy constructor.
   */
  Heap(const Heap& o) : heap(o.heap) {
    /* update fiber usage counts */
    for (auto o : heap) {
      o->use();
    }
    /// @todo For multiple copies of the same fiber (common use case), could
    /// update usage counts just once
  }

  /**
   * Get an allocation.
   *
   * @param index The heap index.
   *
   * @return The raw pointer at the heap index.
   */
  Object* get(const size_t index) {
    assert(index < heap.size());
    assert(heap[index]->getIndex() == index);
    return heap[index];
  }

  /**
   * Set an allocation.
   *
   * @param raw The raw pointer
   *
   * @return Index on the heap.
   */
  void set(const size_t index, Object* raw) {
    raw->setIndex(index);
    heap[index] = raw;
  }

  /**
   * Add a new allocation.
   *
   * @param raw The raw pointer
   *
   * @return Index on the heap.
   */
  size_t put(Object* raw) {
    heap.push_back(raw);
    size_t index = heap.size() - 1;
    raw->setIndex(index);
    return index;
  }

private:
  /**
   * Allocations.
   */
  std::vector<Object*,traceable_allocator<Object*>> heap;
};
}
