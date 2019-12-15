/**
 * @file
 */
#pragma once

#include "libbirch/class.hpp"

namespace libbirch {
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
  using value_type = T;

  /**
   * Constructor. This is intended for use immediately after construction of
   * the object; the reference count is not incremented, as it should be
   * initialized accordingly.
   */
  explicit SharedPtr(T* ptr = nullptr) :
      ptr(ptr) {
    if (ptr) {
      ptr->init();
    }
  }

  /**
   * Constructor with in-place construction of the referent.
   */
  template<class... Args>
  SharedPtr(Args... args) :
      ptr(new T(args...)) {
    //
  }

  /**
   * Generic weak constructor.
   */
  template<class U>
  SharedPtr(const WeakPtr<U>& o) :
      ptr(o.ptr ? static_cast<T*>(o.ptr) : nullptr) {
    if (ptr) {
      assert(ptr->numShared() > 0);
      ptr->incShared();
    }
  }

  /**
   * Generic init constructor.
   */
  template<class U>
  SharedPtr(const InitPtr<U>& o) :
      ptr(o.ptr ? static_cast<T*>(o.ptr) : nullptr) {
    if (ptr) {
      assert(ptr->numShared() > 0);
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
   * Generic copy constructor.
   */
  template<class U>
  SharedPtr(const SharedPtr<U>& o) :
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
   * Generic move constructor.
   */
  template<class U>
  SharedPtr(SharedPtr<U> && o) :
      ptr(o.ptr) {
    o.ptr = nullptr;
  }

  /**
   * Destructor.
   */
  ~SharedPtr() {
    release();
  }

  /**
   * Copy assignment.
   */
  SharedPtr<T>& operator=(const SharedPtr<T>& o) {
    if (o.ptr) {
      o.ptr->incShared();
    }
    auto old = ptr;
    ptr = o.ptr;
    if (old) {
      old->decShared();
    }
    return *this;
  }

  /**
   * Generic copy assignment.
   */
  template<class U>
  SharedPtr<T>& operator=(const SharedPtr<U>& o) {
    if (o.ptr) {
      o.ptr->incShared();
    }
    auto old = ptr;
    ptr = o.ptr;
    if (old) {
      old->decShared();
    }
    return *this;
  }

  /**
   * Move assignment.
   */
  SharedPtr<T>& operator=(SharedPtr<T> && o) {
    auto old = ptr;
    ptr = o.ptr;
    o.ptr = nullptr;
    if (old) {
      old->decShared();
    }
    return *this;
  }

  /**
   * Generic move assignment.
   */
  template<class U>
  SharedPtr<T>& operator=(SharedPtr<U> && o) {
    auto old = ptr;
    ptr = o.ptr;
    o.ptr = nullptr;
    if (old) {
      old->decShared();
    }
    return *this;
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
  T* pull() const {
    assert(!ptr || ptr->numShared() > 0);
    return ptr;
  }

  /**
   * Replace.
   */
  void replace(T* ptr) {
    //assert(!ptr || ptr->numShared() > 0);
    auto old = this->ptr;
    if (ptr) {
      ptr->incShared();
    }
    this->ptr = ptr;
    if (old) {
      old->decShared();
    }
  }

  /**
   * Release.
   */
  void release() {
    if (ptr) {
      ptr->decShared();
      ptr = nullptr;
    }
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
   * Dynamic cast.
   */
  template<class U>
  auto dynamic_pointer_cast() const {
    U cast;
    cast.replace(dynamic_cast<typename U::value_type*>(ptr));
    return cast;
  }

  /**
   * Static cast.
   */
  template<class U>
  auto static_pointer_cast() const {
    U cast;
    cast.replace(static_cast<typename U::value_type*>(ptr));
    return cast;
  }

private:
  /**
   * Raw pointer.
   */
  T* ptr;
};
}
