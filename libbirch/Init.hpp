/**
 * @file
 */
#pragma once

#include "libbirch/Any.hpp"
#include "libbirch/type.hpp"

namespace libbirch {
/**
 * Pointer that initializes to `nullptr`.
 *
 * @ingroup libbirch
 *
 * @tparam T Type, must derive from Any.
 */
template<class T>
class Init {
  template<class U> friend class Shared;
  template<class U> friend class Weak;
  template<class U> friend class Init;
  template<class U> friend class Lazy;
public:
  using value_type = T;

  /**
   * Constructor.
   */
  explicit Init(value_type* ptr = nullptr) :
      ptr(ptr) {
    //
  }

  /**
   * Constructor.
   */
  template<class Q, class U = typename Q::value_type,
      std::enable_if_t<std::is_base_of<T,U>::value,int> = 0>
  Init(const Q& o) :
      ptr(o.ptr) {
    //
  }

  /**
   * Correctly initialize after a bitwise copy.
   */
  void bitwiseFix() {
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
  T* ptr;
};

template<class T>
struct is_value<Init<T>> {
  static const bool value = false;
};

template<class T>
struct is_pointer<Init<T>> {
  static const bool value = true;
};

template<class T>
struct raw<Init<T>> {
  using type = T*;
};
}
