/**
 * @file
 */
#pragma once

#include "libbirch/thread.hpp"

namespace libbirch {
/**
 * STL-compatible allocator that wraps procedural implementation of the
 * pooled allocator.
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

  static T* allocate(const size_t n);

  static T* reallocate(T* ptr1, const size_t n1, const size_t n2);

  static void deallocate(T* ptr, const size_t n);

  bool operator==(const Allocator<T>& o) const {
    return true;
  }

  bool operator!=(const Allocator<T>& o) const {
    return false;
  }
};
}

#include "libbirch/memory.hpp"

template<class T>
T* libbirch::Allocator<T>::allocate(const size_t n) {
  return static_cast<T*>(libbirch::allocate(n * sizeof(T)));
}

template<class T>
T* libbirch::Allocator<T>::reallocate(T* ptr1, const size_t n1, const size_t n2) {
  return static_cast<T*>(libbirch::reallocate(ptr1, n1 * sizeof(T), omp_get_thread_num(), n2 * sizeof(T)));
}

template<class T>
void libbirch::Allocator<T>::deallocate(T* ptr, const size_t n) {
  libbirch::deallocate(ptr, n * sizeof(T), omp_get_thread_num());
}
