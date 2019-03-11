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
  template<class U> using cast_type = SharedPtr<U>;

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
      assert(ptr->numShared() > 0);
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
    ptr = nullptr;
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

private:
  /**
   * Raw pointer.
   */
  T* ptr;
};
}
