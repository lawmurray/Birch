/**
 * @file
 */
#pragma once

#include "libbirch/Any.hpp"
#include "libbirch/Atomic.hpp"
#include "libbirch/type.hpp"

namespace libbirch {
/**
 * Weak pointer with intrusive implementation.
 *
 * @ingroup libbirch
 *
 * @tparam T Type, must derive from Any.
 */
template<class T>
class WeakPtr {
  template<class U> friend class SharedPtr;
  template<class U> friend class WeakPtr;
  template<class U> friend class InitPtr;
public:
  using value_type = T;

  /**
   * Constructor.
   */
  explicit WeakPtr(value_type* ptr = nullptr) :
      ptr(ptr) {
    if (ptr) {
      ptr->incWeak();
    }
  }

  /**
   * Copy constructor.
   */
  WeakPtr(const WeakPtr& o) {
    auto ptr = o.ptr.load();
    if (ptr) {
      ptr->incWeak();
    }
    this->ptr.store(ptr);
  }

  /**
   * Generic copy constructor.
   */
  template<class Q, class U = typename Q::value_type,
      std::enable_if_t<std::is_base_of<T,U>::value,int> = 0>
  WeakPtr(const Q& o) {
    auto ptr = o.ptr.load();
    if (ptr) {
      ptr->incWeak();
    }
    this->ptr.store(ptr);
  }

  /**
   * Move constructor.
   */
  WeakPtr(WeakPtr&& o) {
    ptr.store(o.ptr.exchange(nullptr));
  }

  /**
   * Generic move constructor.
   */
  template<class U, std::enable_if_t<std::is_base_of<T,U>::value,int> = 0>
  WeakPtr(WeakPtr<U>&& o) {
    ptr.store(o.ptr.exchange(nullptr));
  }

  /**
   * Destructor.
   */
  ~WeakPtr() {
    release();
  }

  /**
   * Copy assignment.
   */
  WeakPtr& operator=(const WeakPtr& o) {
    replace(o.ptr.load());
    return *this;
  }

  /**
   * Generic copy assignment.
   */
  template<class Q, class U = typename Q::value_type,
      std::enable_if_t<std::is_base_of<T,U>::value,int> = 0>
  WeakPtr& operator=(const Q& o) {
    replace(o.ptr.load());
    return *this;
  }

  /**
   * Move assignment.
   */
  WeakPtr& operator=(WeakPtr&& o) {
    auto ptr = o.ptr.exchange(nullptr);
    auto old = this->ptr.exchange(ptr);
    if (old) {
      old->decWeak();
    }
    return *this;
  }

  /**
   * Generic move assignment.
   */
  template<class U, std::enable_if_t<std::is_base_of<T,U>::value,int> = 0>
  WeakPtr& operator=(WeakPtr<U>&& o) {
    auto ptr = o.ptr.exchange(nullptr);
    auto old = this->ptr.exchange(ptr);
    if (old) {
      old->decWeak();
    }
    return *this;
  }

  /**
   * Is the pointer not null?
   *
   * This is used instead of an `operator bool()` so as not to conflict with
   * conversion operators in the referent type.
   */
  bool query() const {
    return ptr.load() != nullptr;
  }

  /**
   * Get the raw pointer.
   */
  T* get() const {
    return ptr.load();
  }

  /**
   * Get the raw pointer as const.
   */
  T* pull() const {
    return ptr.load();
  }

  /**
   * Replace.
   */
  void replace(T* ptr) {
    auto old = this->ptr.exchange(ptr);
    if (ptr) {
      ptr->incWeak();
    }
    if (old) {
      old->decWeak();
    }
  }

  /**
   * Release.
   */
  void release() {
    auto old = ptr.exchange(nullptr);
    if (old) {
      old->decWeak();
    }
  }

  /**
   * Discard.
   */
  void discard() {
    // nothing to do for weak pointers
  }

  /**
   * Restore.
   */
  void restore() {
    // nothing to do for weak pointers
  }

  /**
   * Has this been discarded?
   */
  static bool isDiscarded() {
    return false;
  }

  /**
   * Dereference.
   */
  T& operator*() const {
    return *get();
  }

  /**
   * Member access.
   */
  T* operator->() const {
    return get();
  }

private:
  /**
   * Raw pointer.
   */
  Atomic<T*> ptr;
};

template<class T>
struct is_value<WeakPtr<T>> {
  static const bool value = false;
};

template<class T>
struct is_pointer<WeakPtr<T>> {
  static const bool value = true;
};

template<class T>
struct raw<WeakPtr<T>> {
  using type = T*;
};
}
