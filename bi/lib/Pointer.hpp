/**
 * @file
 */
#pragma once

#include <cstddef>

namespace bi {
/**
 * Smart copy-on-write pointer.
 *
 * @ingroup library
 *
 * @tparam T Type.
 */
template<class T>
class Pointer {
  template<class U> friend class Pointer;
public:
  /**
   * Constructor.
   */
  Pointer(T* ptr = nullptr) : ptr(ptr) {
    //
  }

  /**
   * Generic constructor.
   */
  template<class U>
  Pointer(U* ptr = nullptr) : ptr(ptr) {
    //
  }

  /**
   * Copy constructor.
   */
  Pointer(const Pointer<T>& o) : ptr(o.ptr) {
    //
  }

  /**
   * Generic copy constructor.
   */
  template<class U>
  Pointer(const Pointer<U>& o) : ptr(o.ptr) {
    //
  }

  /**
   * Get the raw pointer.
   */
  T* get() {
    if (ptr->isShared()) {
      /* shared and writeable, so copy now (copy-on-write) */
      auto from = ptr;
      auto to = dynamic_cast<T*>(from->clone());
      from->disuse();
      to->use();

      /* update other uses of this pointer in the same coroutine */
      //coroutine->replace(from, to);  ///@todo

      return to;
    } else {
      /* not shared, so no need to copy */
      return ptr;
    }
  }
  T* const get() const {
    /* read-only, so no need to copy */
    return ptr;
  }

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

  /*
   * Equality operators.
   */
  bool operator==(const Pointer<T>& o) const {
    return get() == o.get();
  }
  bool operator!=(const Pointer<T>& o) const {
    return !(*this == o);
  }

private:
  /**
   * Raw pointer.
   */
  T* ptr;
};
}
