/**
 * @file
 */
#pragma once

#include <vector>

namespace bi {
class Object;

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
  Heap();

  /**
   * Copy constructor.
   */
  Heap(const Heap& o);

  /**
   * Move constructor.
   */
  Heap(Heap&& o) = default;

  /**
   * Destructor.
   */
  virtual ~Heap();

  /**
   * Copy assignment. It is common for fibers to have shared history, and an
   * assignment provides the opportunity to only update usage counts for the
   * part that is not shared, which may be faster than a complete destruction
   * and recreation.
   */
  Heap& operator=(const Heap& o);

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
  Object* get(const size_t index);

  /**
   * Set an allocation.
   *
   * @param raw The raw pointer
   *
   * @return Index on the heap.
   */
  void set(const size_t index, Object* raw);

  /**
   * Add a new allocation.
   *
   * @param raw The raw pointer
   *
   * @return Index on the heap.
   */
  size_t put(Object* raw);

private:
  /**
   * Allocations.
   */
  std::vector<Object*> heap;
};
}
