/**
 * @file
 */
#pragma once

#include "libbirch/SharedPtr.hpp"

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
   * Constructor.
   */
  WeakPtr(const SharedPtr<T>& o) :
      ptr(o.ptr) {
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
      ptr = nullptr;
    }
  }

  /**
   * Copy assignment.
   */
  WeakPtr<T>& operator=(const WeakPtr<T>& o) {
    auto old = ptr;
    ptr = o.ptr;
    if (ptr) {
      ptr->incWeak();
    }
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
   * Obtain a shared pointer if the object still exists, otherwise a
   * null pointer.
   */
  SharedPtr<T> lock() const {
    if (ptr) {
      return ptr->template lock<T>();
    } else {
      return nullptr;
    }
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
