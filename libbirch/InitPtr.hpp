/**
 * @file
 */
#pragma once

#include "libbirch/class.hpp"
#include "libbirch/Counted.hpp"

namespace libbirch {
/**
 * Smart pointer that does not update reference counts, but that does
 * initialize to nullptr.
 *
 * @ingroup libbirch
 *
 * @tparam T Type, must derive from Counted.
 */
template<class T>
class InitPtr: public InitPtr<typename bi::type::super_type<T>::type> {
public:
  using value_type = T;
  using super_type = InitPtr<typename bi::type::super_type<value_type>::type>;
  using this_type = InitPtr<value_type>;

  /**
   * Constructor.
   */
  explicit InitPtr(value_type* ptr = nullptr) :
      super_type(ptr) {
    //
  }

  /**
   * Constructor.
   */
  template<class Q, std::enable_if_t<!std::is_pointer<Q>::value && is_base_of<this_type,Q>::value,int> = 0>
  InitPtr(const Q& o) :
      super_type(o) {
    //
  }

  /**
   * Get the raw pointer.
   */
  T* get() const {
    return static_cast<T*>(super_type::get());
  }

  /**
   * Get the raw pointer as const.
   */
  T* pull() const {
    return static_cast<T*>(super_type::pull());
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
};

/**
 * Smart pointer that does not update reference counts, but that does
 * initialize to nullptr.
 *
 * @ingroup libbirch
 */
template<>
class InitPtr<Counted> {
public:
  using value_type = Counted;
  using this_type = InitPtr<Counted>;

  /**
   * Constructor.
   */
  explicit InitPtr(value_type* ptr = nullptr) : ptr(ptr) {
    //
  }

  /**
   * Constructor.
   */
  template<class Q, std::enable_if_t<!std::is_pointer<Q>::value && is_base_of<this_type,Q>::value,int> = 0>
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
  Counted* get() const {
    return ptr;
  }

  /**
   * Get the raw pointer as const.
   */
  Counted* pull() const {
    return ptr;
  }

  /**
   * Replace.
   */
  void replace(Counted* ptr) {
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
  Counted& operator*() const {
    return *get();
  }

  /**
   * Member access.
   */
  Counted* operator->() const {
    return get();
  }

  /**
   * Equal comparison.
   */
  template<class Q>
  bool operator==(const Q& o) const {
    return get() == o.get();
  }

  /**
   * Not equal comparison.
   */
  template<class Q>
  bool operator!=(const Q& o) const {
    return get() != o.get();
  }

private:
  /**
   * Raw pointer.
   */
  Counted* ptr;
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
struct raw_type<InitPtr<T>> {
  using type = T*;
};
}
