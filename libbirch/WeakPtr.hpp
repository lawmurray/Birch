/**
 * @file
 */
#pragma once

#include "libbirch/config.hpp"

namespace bi {
template<class T> class SharedPtr;
template<class T> class WeakPtr;

/**
 * Weak pointer with intrusive implementation.
 *
 * @ingroup libbirch
 *
 * @tparam T Type, must derive from Counted.
 */
template<class T>
class WeakPtr {
  template<class U> friend class SharedPtr;
  template<class U> friend class WeakPtr;
public:
  /**
   * Constructor.
   */
  WeakPtr(T* ptr = nullptr) :
      ptr(ptr) {
    if (ptr) {
      ptr->incWeak();
    }
  }

  /**
   * Copy constructor.
   */
  WeakPtr(const WeakPtr<T>& o) :
      ptr(o.ptr) {
    if (ptr) {
      assert(ptr->numWeak() > 0);
      ptr->incWeak();
    }
  }

  /**
   * Copy constructor.
   */
  WeakPtr(const SharedPtr<T>& o) :
      ptr(o.ptr) {
    if (ptr) {
      assert(ptr->numWeak() > 0);
      ptr->incWeak();
    }
  }

  /**
   * Move constructor.
   */
  WeakPtr(WeakPtr<T> && o) :
      ptr(o.ptr) {
    o.ptr = nullptr;
  }

  /**
   * Destructor.
   */
  ~WeakPtr() {
    if (ptr) {
      ptr->decWeak();
    }
    #ifndef NDEBUG
    ptr = nullptr;
    #endif
  }

  /**
   * Copy assignment.
   */
  WeakPtr<T>& operator=(const WeakPtr<T>& o) {
    if (o.ptr) {
      o.ptr->incWeak();
    }
    auto old = ptr;
    ptr = o.ptr;
    if (old) {
      old->decWeak();
    }
    return *this;
  }

  /**
   * Move assignment.
   */
  WeakPtr<T>& operator=(WeakPtr<T> && o) {
    std::swap(ptr, o.ptr);
    return *this;
  }

  /**
   * Get the raw pointer.
   */
  T* get() const {
    return ptr;
  }

  /**
   * Equal comparison.
   */
  bool operator==(const WeakPtr<T>& o) const {
    return ptr == o.ptr;
  }

  /**
   * Not equal comparison.
   */
  bool operator!=(const WeakPtr<T>& o) const {
    return ptr != o.ptr;
  }

  /**
   * Is the pointer not null?
   */
  operator bool() const {
    return ptr != nullptr;
  }

private:
  /**
   * Raw pointer.
   */
  T* ptr;
};
}
