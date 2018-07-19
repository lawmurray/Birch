/**
 * @file
 */
#pragma once

#include "libbirch/global.hpp"

#if __cplusplus > 201703L
#include <new>
#endif

namespace bi {
/**
 * Heap allocator for objects of class type. Maintains a pool of allocations,
 * rounding sizes up to the nearest power of two for efficient reuse.
 *
 * @ingroup libbirch
 *
 * @tparam T Value type.
 */
template<class T>
class Allocator {
public:
  using value_type = T;

  Allocator() = default;
  Allocator(const Allocator<T>&) = default;
  Allocator(Allocator<T> &&) = default;

  template<class U>
  Allocator(const Allocator<U>& o) {
    //
  }

  static value_type* allocate(const size_t n) {
    return static_cast<value_type*>(bi::allocate(n * sizeof(value_type)));
  }

  static value_type* reallocate(value_type* ptr1, const size_t n1,
      const size_t n2) {
    return static_cast<value_type*>(bi::reallocate(ptr1,
        n1 * sizeof(value_type), n2 * sizeof(value_type)));
  }

  static void deallocate(value_type* ptr, const size_t n) {
    bi::deallocate(ptr, n * sizeof(value_type));
  }

  bool operator==(const Allocator<T>& o) const {
    return true;
  }

  bool operator!=(const Allocator<T>& o) const {
    return false;
  }
};
}
