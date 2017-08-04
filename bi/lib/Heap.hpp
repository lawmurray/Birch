/**
 * @file
 */
#pragma once

#include "bi/lib/Object.hpp"

#include <vector>
#include <gc/gc_allocator.h>

#include <iostream>

namespace bi {
/**
 * Heap-local heap.
 *
 * @ingroup library
 */
class Heap {
public:
  /**
   * Constructor.
   */
  Heap() {
    //
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
   * Move constructor.
   */
  Heap(Heap&& o) = default;

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
   * Copy assignment. It is common for fibers to have shared history, and an
   * assignment provides the opportunity to only update usage counts for the
   * part that is not shared, which may be faster than a complete destruction
   * and recreation.
   */
  Heap& operator=(const Heap& o) {
    auto iter1 = heap.begin();
    auto iter2 = o.heap.begin();
    auto end1 = heap.end();
    auto end2 = o.heap.end();

    /* skip through common history */
    while (iter1 != end1 && iter2 != end2 && *iter1 == *iter2) {
      ++iter1;
      ++iter2;
    }

    /* disuse remaining allocations on the left of the assignment */
    while (iter1 != end1) {
      (*iter1)->disuse();
      ++iter1;
    }

    /* copy remaining allocations from the right side of the assignment */
    heap.resize(o.heap.size());
    iter1 = heap.begin() + std::distance(o.heap.begin(), iter2);
    while (iter2 != end2) {
      *iter1 = *iter2;
      (*iter1)->use();
      ++iter1;
      ++iter2;
    }
    assert(iter1 == heap.end());
    assert(iter2 == o.heap.end());

    return *this;
  }

  /**
   * Move assignment.
   */
  Heap& operator=(Heap&& o) = default;

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
