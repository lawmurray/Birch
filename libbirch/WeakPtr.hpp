/**
 * @file
 */
#pragma once

#include "libbirch/config.hpp"
#include "libbirch/class.hpp"

namespace bi {
template<class T> class SharedPtr;
template<class T> class WeakPtr;
template<class T> class InitPtr;

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
  template<class U> friend class InitPtr;
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
   * Generic shared constructor.
   */
  template<class U>
  WeakPtr(const SharedPtr<U>& o) :
      ptr(o.ptr) {
    if (ptr) {
      ptr->incWeak();
    }
  }

  /**
   * Generic weak constructor.
   */
  template<class U>
  WeakPtr(const WeakPtr<U>& o) :
      ptr(o.ptr) {
    if (ptr) {
      ptr->incWeak();
    }
  }

  /**
   * Generic init constructor.
   */
  template<class U>
  WeakPtr(const InitPtr<U>& o) :
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
    }
  }

  /**
   * Copy assignment.
   */
  WeakPtr<T>& operator=(const WeakPtr<T>& o) {
    if (ptr != o.ptr) {
      if (o.ptr) {
        o.ptr->incWeak();
      }
      auto old = ptr;
      ptr = o.ptr;
      if (old) {
        old->decWeak();
      }
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
   * Value assignment.
   */
  template<class U,
      typename = std::enable_if_t<bi::has_assignment<T,U>::value>>
  WeakPtr<T>& operator=(const U& o) {
    *ptr = o;
    return *this;
  }

  /**
   * Value conversion.
   */
  template<class U,
      typename = std::enable_if_t<bi::has_conversion<T,U>::value>>
  operator U() const {
    return static_cast<U>(*ptr);
  }

  /**
   * Is the pointer not null?
   */
  bool query() const {
    return ptr != nullptr;
  }

  /**
   * Get the raw pointer.
   */
  T* get() const {
    assert(!ptr || ptr->numShared() > 0);
    return ptr;
  }

  /**
   * Get the raw pointer as const.
   */
  const T* pull() const {
    assert(!ptr || ptr->numShared() > 0);
    return ptr;
  }

  /**
   * Dereference.
   */
  T& operator*() const {
    assert(ptr);
    assert(ptr->numShared() > 0);
    return *ptr;
  }

  /**
   * Member access.
   */
  T* operator->() const {
    assert(ptr);
    assert(ptr->numShared() > 0);
    return ptr;
  }

  /**
   * Equal comparison.
   */
  bool operator==(const SharedPtr<T>& o) const {
    return ptr == o.ptr;
  }
  bool operator==(const WeakPtr<T>& o) const {
    return ptr == o.ptr;
  }
  bool operator==(const InitPtr<T>& o) const {
    return ptr == o.ptr;
  }
  bool operator==(const T* o) const {
    return ptr == o;
  }

  /**
   * Not equal comparison.
   */
  bool operator!=(const SharedPtr<T>& o) const {
    return ptr != o.ptr;
  }
  bool operator!=(const WeakPtr<T>& o) const {
    return ptr != o.ptr;
  }
  bool operator!=(const InitPtr<T>& o) const {
    return ptr != o.ptr;
  }
  bool operator!=(const T* o) const {
    return ptr != o;
  }

  /**
   * Is the pointer not null?
   */
  operator bool() const {
    return ptr != nullptr;
  }

  /**
   * Dynamic cast. Returns `nullptr` if unsuccessful.
   */
  template<class U>
  WeakPtr<U> dynamic_pointer_cast() const {
    return SharedPtr<U>(dynamic_cast<U*>(ptr));
  }

  /**
   * Static cast. Undefined if unsuccessful.
   */
  template<class U>
  WeakPtr<U> static_pointer_cast() const {
    return SharedPtr<U>(static_cast<U*>(ptr));
  }

private:
  /**
   * Raw pointer.
   */
  T* ptr;
};
}
