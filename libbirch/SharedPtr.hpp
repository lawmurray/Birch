/**
 * @file
 */
#pragma once

#include "libbirch/Any.hpp"
#include "libbirch/type.hpp"

namespace libbirch {
/**
 * Shared pointer with intrusive implementation.
 *
 * @ingroup libbirch
 *
 * @tparam T Type, must derive from Any.
 */
template<class T>
class SharedPtr {
public:
  using value_type = T;

  /**
   * Constructor.
   */
  explicit SharedPtr(value_type* ptr = nullptr) :
      ptr(ptr) {
    if (ptr) {
      ptr->incShared();
    }
  }

  /**
   * Constructor.
   */
  template<class Q, class U = typename Q::value_type,
      std::enable_if_t<std::is_base_of<T,U>::value,int> = 0>
  SharedPtr(const Q& o) :
      ptr(o.get()) {
    if (ptr) {
      ptr->incShared();
    }
  }

  /**
   * Copy constructor.
   */
  SharedPtr(const SharedPtr& o) :
      ptr(o.ptr) {
    if (ptr) {
      assert(ptr->numShared() > 0u);
      ptr->incShared();
    }
  }

  /**
   * Move constructor.
   */
  SharedPtr(SharedPtr&& o) :
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
  SharedPtr& operator=(const SharedPtr& o) {
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
  SharedPtr& operator=(SharedPtr&& o) {
    auto old = ptr;
    ptr = o.ptr;
    o.ptr = nullptr;
    if (old) {
      old->decShared();
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
  T* pull() const {
    assert(!ptr || ptr->numShared() > 0);
    return ptr;
  }

  /**
   * Replace.
   */
  void replace(T* ptr) {
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
  T* ptr;

};

template<class T>
struct is_value<SharedPtr<T>> {
  static const bool value = false;
};

template<class T>
struct is_pointer<SharedPtr<T>> {
  static const bool value = true;
};

template<class T>
struct raw<SharedPtr<T>> {
  using type = T*;
};
}
