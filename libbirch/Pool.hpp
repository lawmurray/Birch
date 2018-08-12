/**
 * @file
 */
#pragma once

#include "libbirch/Lockable.hpp"

#include <vector>
#include <stack>

namespace bi {
/**
 * Thread-safe stack of memory allocations.
 *
 * @ingroup libbirch
 */
class Pool: public Lockable {
public:
  /**
   * Pop an allocation from the pool. Returns `nullptr` if the pool is
   * empty.
   */
  void* pop() {
    set();
    void* result = nullptr;
    if (!pool.empty()) {
      result = pool.top();
      pool.pop();
    }
    unset();
    return result;
  }

  /**
   * Push an allocation to the pool.
   */
  void push(void* ptr) {
    set();
    pool.push(ptr);
    unset();
  }

private:
  /**
   * Stack of allocations.
   */
  std::stack<void*,std::vector<void*>> pool;
};
}
