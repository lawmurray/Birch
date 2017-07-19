/**
 * @file
 */
#pragma once

#include <cstddef>

namespace bi {
/**
 * Relocatable pointer.
 *
 * @ingroup library
 *
 * @tparam T Type.
 */
template<class T>
class Pointer {
  template<class U> friend class Pointer;
  friend class Heap;
public:
  /**
   * Constructor for global pointer.
   */
  explicit Pointer(T* ptr = nullptr) : ptr(ptr), index(-1) {
    //
  }

  /**
   * Generic constructor for global pointer.
   */
  template<class U>
  explicit Pointer(U* ptr = nullptr) : ptr(ptr), index(-1) {
    //
  }

  /**
   * Constructor for coroutine-local pointer.
   */
  explicit Pointer(const size_t index) : ptr(nullptr), index(index) {
    //
  }

  /**
   * Copy constructor.
   */
  Pointer(const Pointer<T>& o) : ptr(o.ptr), index(o.index) {
    //
  }

  /**
   * Generic copy constructor.
   */
  template<class U>
  Pointer(const Pointer<U>& o) : ptr(o.ptr), index(o.index) {
    //
  }

  /*
   * Equality operators.
   */
  bool operator==(const Pointer<T>& o) const {
    return get() == o.get();
  }
  bool operator!=(const Pointer<T>& o) const {
    return !(*this == o);
  }

  /**
   * Get the raw pointer.
   */
  T* get();
  T* const get() const;

  /**
   * Dereference.
   */
  T& operator*() {
    return *get();
  }
  const T& operator*() const {
    return *get();
  }

  /**
   * Member access.
   */
  T* operator->() {
    return get();
  }
  T* const operator->() const {
    return get();
  }

  /**
   * Call operator.
   */
  template<class... Args>
  auto operator()(Args... args) const {
    return (*ptr)(args...);
  }

private:
  /**
   * Raw pointer, for global pointers.
   */
  T* ptr;

  /**
   * Heap index, for coroutine-local pointers.
   */
  size_t index;
};
}

#include "bi/lib/Heap.hpp"

template<class T>
T* bi::Pointer<T>::get() {
  return heap.get(*this);
}

template<class T>
T* const bi::Pointer<T>::get() const {
  return heap.get(*this);
}
