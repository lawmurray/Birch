/**
 * @file
 */
#pragma once

#include "libbirch/Nil.hpp"
#include "libbirch/Shared.hpp"
#include "libbirch/Weak.hpp"
#include "libbirch/Init.hpp"
#include "libbirch/type.hpp"

namespace libbirch {
/**
 * Optional.
 *
 * @ingroup libbirch
 *
 * @tparam T Type.
 */
template<class T, class Enable = void>
class Optional {
  //
};

/**
 * Optional for non-pointer types.
 *
 * @ingroup libbirch
 *
 * @tparam T Non-pointer type.
 */
template<class T>
class Optional<T,std::enable_if_t<!is_pointer<T>::value>> {
public:
  /**
   * Default constructor.
   */
  Optional() :
      value(),
      hasValue(false) {
    //
  }

  /**
   * Constructor for no value.
   */
  Optional(const Nil&) :
      value(),
      hasValue(false) {
    //
  }

  /**
   * Constructor for a value.
   */
  Optional(const T& value) :
      value(value),
      hasValue(true) {
    //
  }

  /**
   * Generic value constructor.
   *
   * @tparam U Value type (convertible to @p T).
   */
  template<class U>
  Optional(const U& value) :
      value(value),
      hasValue(true) {
    //
  }

  /**
   * Generic copy constructor.
   *
   * @tparam U Value type (convertible to @p T).
   */
  template<class U>
  Optional(const Optional<U>& o) :
      hasValue(o.query()) {
    if (hasValue) {
      value = o.get();
    }
  }

  Optional(const Optional<T>& o) = default;
  Optional(Optional<T> && o) = default;
  Optional<T>& operator=(const Optional<T>& o) = default;
  Optional<T>& operator=(Optional<T> && o) = default;

  /**
   * Is there a value?
   */
  bool query() const {
    return hasValue;
  }

  /**
   * Get the value.
   */
  T& get() {
    libbirch_assert_msg_(hasValue, "optional has no value");
    return value;
  }

  /**
   * Get the value.
   */
  const T& get() const {
    libbirch_assert_msg_(hasValue, "optional has no value");
    return value;
  }

private:
  /**
   * The contained value, if any.
   */
  T value;

  /**
   * Is there a value?
   */
  bool hasValue;
};

/**
 * Optional for pointer types. Uses the pointer itself, set to `nullptr`, to
 * denote a missing value, rather than keeping a separate boolean flag.
 *
 * @ingroup libbirch
 *
 * @tparam P Pointer type.
 */
template<class P>
class Optional<P,std::enable_if_t<is_pointer<P>::value>> {
  template<class Q, class Enable> friend class Optional;
public:
  Optional() = default;
  Optional(const Optional<P>& o) = default;
  Optional(Optional<P>&& o) = default;
  Optional<P>& operator=(const Optional<P>& o) = default;
  Optional<P>& operator=(Optional<P>&& o) = default;

  /**
   * Generic conversion constructor.
   */
  template<class Q, typename = std::enable_if_t<is_pointer<Q>::value>>
  Optional(const Optional<Q>& o) : value(o.value) {
    //
  }

  /**
   * Nil constructor.
   */
  Optional(const Nil&) {
    //
  }

  /**
   * Generic value copy constructor.
   */
  template<class Q, typename = std::enable_if_t<is_pointer<Q>::value>>
  Optional(const Q& value) :
      value(value) {
    //
  }

  /**
   * Generic value move constructor.
   */
  template<class Q, typename = std::enable_if_t<is_pointer<Q>::value>>
  Optional(Q&& value) :
      value(std::move(value)) {
    //
  }

  /**
   * Nil assignment.
   */
  Optional<P>& operator=(const Nil& o) {
    value = nullptr;
    return *this;
  }

  /**
   * Generic value copy assignment.
   */
  template<class Q, typename = std::enable_if_t<is_pointer<Q>::value>>
  Optional<P>& operator=(const Q& value) {
    this->value = value;
    return *this;
  }

  /**
   * Generic value move assignment.
   */
  template<class Q, typename = std::enable_if_t<is_pointer<Q>::value>>
  Optional<P>& operator=(Q&& value) {
    this->value = std::move(value);
    return *this;
  }

  /**
   * Is there a value?
   */
  bool query() const {
    return value.query();
  }

  /**
   * Get the value.
   */
  P& get() {
    libbirch_assert_msg_(query(), "optional has no value");
    return value;
  }

  /**
   * Get the value.
   */
  const P& get() const {
    libbirch_assert_msg_(query(), "optional has no value");
    return value;
  }

private:
  /**
   * The value, `nullptr` value to denote no value.
   */
  P value;
};
}
