/**
 * @file
 */
#pragma once

#include "libbirch/Any.hpp"
#include "libbirch/type.hpp"

namespace libbirch {
/**
 * Smart pointer that does not update reference counts, but that does
 * initialize to nullptr.
 *
 * @ingroup libbirch
 *
 * @tparam T Type, must derive from Any.
 */
template<class T>
class InitPtr {
public:
  using value_type = T;

  /**
   * Constructor.
   */
  explicit InitPtr(value_type* ptr = nullptr) :
      ptr(ptr) {
    //
  }

  /**
   * Constructor.
   */
  template<class Q, class U = typename Q::value_type,
      std::enable_if_t<std::is_base_of<T,U>::value,int> = 0>
  InitPtr(const Q& o) :
      ptr(o.get()) {
    //
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
    return ptr;
  }

  /**
   * Get the raw pointer as const.
   */
  T* pull() const {
    return ptr;
  }

  /**
   * Replace.
   */
  void replace(T* ptr) {
    this->ptr = ptr;
  }

  /**
   * Release.
   */
  void release() {
    ptr = nullptr;
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
struct is_value<InitPtr<T>> {
  static const bool value = false;
};

template<class T>
struct is_pointer<InitPtr<T>> {
  static const bool value = true;
};

template<class T>
struct raw<InitPtr<T>> {
  using type = T*;
};
}
