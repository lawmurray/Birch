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
      ptr(o.get()) {
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
    this->ptr.store(ptr);
  }

  /**
   * Release.
   */
  void release() {
    ptr.store(nullptr);
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

  /**
   * Mark.
   */
  void mark() {
    // nothing to do here, cannot create cycles
  }

  /**
   * Scan.
   */
  void scan() {
    // nothing to do here, cannot create cycles
  }

  /**
   * Reach.
   */
  void reach() {
    // nothing to do here, cannot create cycles
  }

  /**
   * Collect.
   */
  void collect() {
    // nothing to do here, cannot create cycles
  }

private:
  /**
   * Raw pointer.
   */
  Atomic<T*> ptr;
};

template<class T>
struct is_value<Init<T>> {
  static const bool value = false;
};

template<class T>
struct is_pointer<Init<T>> {
  static const bool value = true;
};

template<class T, unsigned N>
struct is_acyclic<Init<T>,N> {
  static const bool value = true;
};

template<class T>
struct raw<Init<T>> {
  using type = T*;
};
}
