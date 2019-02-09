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
 * Shared pointer with intrusive implementation.
 *
 * @ingroup libbirch
 *
 * @tparam T Type, must derive from Counted.
 */
template<class T>
class SharedPtr {
  template<class U> friend class SharedPtr;
  template<class U> friend class WeakPtr;
  template<class U> friend class InitPtr;
public:
  /**
   * Constructor.
   */
  SharedPtr(T* ptr = nullptr) :
      ptr(ptr) {
    if (ptr) {
      ptr->incShared();
    }
  }

  /**
   * Generic shared constructor.
   */
  template<class U>
  SharedPtr(const SharedPtr<U>& o) :
      ptr(o.ptr) {
    if (ptr) {
      ptr->incShared();
    }
  }

  /**
   * Generic weak constructor.
   */
  template<class U>
  SharedPtr(const WeakPtr<U>& o) :
      ptr(o.ptr ? static_cast<T*>(o.ptr->lock()) : nullptr) {
    //
  }

  /**
   * Generic init constructor.
   */
  template<class U>
  SharedPtr(const InitPtr<U>& o) :
      ptr(o.ptr) {
    if (ptr) {
      ptr->incShared();
    }
  }

  /**
   * Copy constructor.
   */
  SharedPtr(const SharedPtr<T>& o) :
      ptr(o.ptr) {
    if (ptr) {
      ptr->incShared();
    }
  }

  /**
   * Move constructor.
   */
  SharedPtr(SharedPtr<T> && o) :
      ptr(o.ptr) {
    o.ptr = nullptr;
  }

  /**
   * Destructor.
   */
  ~SharedPtr() {
    if (ptr) {
      ptr->decShared();
    }
  }

  /**
   * Copy assignment.
   */
  SharedPtr<T>& operator=(const SharedPtr<T>& o) {
    if (ptr != o.ptr) {
      if (o.ptr) {
        o.ptr->incShared();
      }
      auto old = ptr;
      ptr = o.ptr;
      if (old) {
        old->decShared();
      }
    }
    return *this;
  }

  /**
   * Move assignment.
   */
  SharedPtr<T>& operator=(SharedPtr<T> && o) {
    std::swap(ptr, o.ptr);
    return *this;
  }

  /**
   * Value assignment.
   */
  template<class U,
      typename = std::enable_if_t<bi::has_assignment<T,U>::value>>
  SharedPtr<T>& operator=(const U& o) {
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
   * Get the raw pointer as const;
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
    return WeakPtr<U>(dynamic_cast<U*>(ptr));
  }

  /**
   * Static cast. Undefined if unsuccessful.
   */
  template<class U>
  WeakPtr<U> static_pointer_cast() const {
    return WeakPtr<U>(static_cast<U*>(ptr));
  }

private:
  /**
   * Raw pointer.
   */
  T* ptr;
};
}
