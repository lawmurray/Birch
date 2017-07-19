/**
 * @file
 */
#pragma once

#include "bi/lib/Object.hpp"
#include "bi/lib/Heap.hpp"

namespace bi {
/**
 * Relocatable coroutine.
 *
 * @ingroup library
 *
 * @tparam Type Return type.
 */
template<class Type>
class Coroutine : public Object {
public:
  /**
   * Constructor.
   */
  Coroutine() : state(0) {
    //
  }

  /**
   * Run to next yield.
   */
  virtual Type operator()() = 0;

  /**
   * Run garbage collector on coroutine-local allocations.
   */
  void collect() {
    mark();
    heap.sweep();
  }

protected:
  /**
   * Coroutine-local heap.
   */
  Heap heap;

  /**
   * State.
   */
  int state;
};
}
