/**
 * @file
 */
#pragma once

#include <cstddef>

namespace bi {
/**
 * Smart pointer for global and coroutine-local objects, with copy-on-write
 * semantics for te latter.
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
  Pointer(T* ptr = nullptr, const size_t index = -1) :
      ptr(ptr),
      index(index) {
    //
  }

  /**
   * Generic constructor.
   */
  template<class U>
  Pointer(U* ptr = nullptr, const size_t index = -1) :
      ptr(ptr),
      index(index) {
    //
  }

  /**
   * Copy constructor.
   */
  Pointer(const Pointer<T>& o) = default;

  /**
   * Generic copy constructor.
   */
  template<class U>
  Pointer(const Pointer<U>& o) :
      ptr(o.ptr),
      index(o.index) {
    //
  }

  /**
   * Get the raw pointer.
   */
  T* get() {
    if (ptr->isShared()) {
      /* shared and writeable, copy now (copy-on-write) */
      auto from = ptr;
      auto to = static_cast<T*>(from->clone());
      from->disuse();
      to->use();
      return to;
    } else {
      /* not shared, no need to copy */
      return ptr;
    }
  }
  T* const get() const {
    /* read-only, no need to copy */
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
  template<class ... Args>
  auto operator()(Args ... args) const {
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
   * For a global pointer, the raw address.
   */
  T* ptr;

  /**
   * For a coroutine-local pointer, the index of the heap allocation,
   * otherwise -1.
   */
  size_t index;

  /// @todo Might there be an implementation that allows both cases to be
  /// packed into the same 64-bit value?
};
}
