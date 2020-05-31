/**
 * @file
 */
#pragma once

#include "libbirch/Any.hpp"
#include "libbirch/Atomic.hpp"
#include "libbirch/type.hpp"

namespace libbirch {
/**
 * Weak pointer.
 *
 * @ingroup libbirch
 *
 * @tparam T Type, must derive from Any.
 */
template<class T>
class Weak {
  template<class U> friend class Shared;
  template<class U> friend class Weak;
  template<class U> friend class Init;
  template<class U> friend class Lazy;
public:
  using value_type = T;

  /**
   * Constructor.
   */
  explicit Weak(value_type* ptr = nullptr) :
      ptr(ptr) {
    if (ptr) {
      ptr->incWeak();
    }
  }

  /**
   * Copy constructor.
   */
  Weak(const Weak& o) {
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
  Weak(const Q& o) {
    auto ptr = o.ptr.load();
    if (ptr) {
      ptr->incWeak();
    }
    this->ptr.store(ptr);
  }

  /**
   * Move constructor.
   */
  Weak(Weak&& o) {
    ptr.store(o.ptr.exchange(nullptr));
  }

  /**
   * Generic move constructor.
   */
  template<class U, std::enable_if_t<std::is_base_of<T,U>::value,int> = 0>
  Weak(Weak<U>&& o) {
    ptr.store(o.ptr.exchange(nullptr));
  }

  /**
   * Destructor.
   */
  ~Weak() {
    release();
  }

  /**
   * Fix after a bitwise copy.
   */
  void bitwiseFix() {
    if (ptr.get()) {
      ptr.get()->incWeak();
    }
  }

  /**
   * Copy assignment.
   */
  Weak& operator=(const Weak& o) {
    replace(o.get());
    return *this;
  }

  /**
   * Generic copy assignment.
   */
  template<class Q, class U = typename Q::value_type,
      std::enable_if_t<std::is_base_of<T,U>::value,int> = 0>
  Weak& operator=(const Q& o) {
    replace(o.get());
    return *this;
  }

  /**
   * Move assignment.
   */
  Weak& operator=(Weak&& o) {
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
  Weak& operator=(Weak<U>&& o) {
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
   * Replace.
   */
  void replace(T* ptr) {
    assert(ptr->numMemoShared() > 0u);
    if (ptr) {
      ptr->incWeak();
    }
    auto old = this->ptr.exchange(ptr);
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

private:
  /**
   * Raw pointer.
   */
  Atomic<T*> ptr;
};

template<class T>
struct is_value<Weak<T>> {
  static const bool value = false;
};

template<class T>
struct is_pointer<Weak<T>> {
  static const bool value = true;
};

template<class T>
struct raw<Weak<T>> {
  using type = T*;
};
}
