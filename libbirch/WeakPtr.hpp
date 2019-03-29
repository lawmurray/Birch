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
  using value_type = T;
  template<class U> using cast_type = WeakPtr<U>;

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
      assert(ptr->numWeak() > 0);
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
    ptr = nullptr;
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
   * Get the raw pointer.
   */
  T* get() const {
    assert(!ptr || ptr->numWeak() > 0);
    return ptr;
  }

  /**
   * Get the raw pointer as const.
   */
  const T* pull() const {
    assert(!ptr || ptr->numWeak() > 0);
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
    assert(ptr->numWeak() > 0);
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
