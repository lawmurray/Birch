/**
 * @file
 */
#pragma once

#include "libbirch/type.hpp"
#include "libbirch/Nil.hpp"
#include "libbirch/Lazy.hpp"

namespace libbirch {
/**
 * Optional.
 *
 * @ingroup libbirch
 *
 * @tparam T Value type.
 */
template<class T, class Enable = void>
class Optional {
  //
};

/**
 * Optional for non-array and non-pointer types.
 *
 * @ingroup libbirch
 *
 * @tparam T Value type.
 */
template<class T>
class Optional<T,std::enable_if_t<!is_array<T>::value && !is_pointer<T>::value>> {
  template<class U, class Enable1> friend class Optional;

  static_assert(!std::is_lvalue_reference<T>::value,
      "Optional does not support lvalue reference types.");
public:
  /**
   * Constructor.
   */
  Optional(const Nil& = nil) :
      value(),
      hasValue(false) {
    //
  }

  /**
   * Value copy constructor. A template is used to ensure that only a value
   * of type T can be implicitly converted to a value of Optional<T>.
   * Implicit type conversions, especially numerical conversions, otherwise
   * cause troublesome ambiguous functions calls when passing a value
   * argument to an optional parameter.
   */
  template<class U, IS_SAME(T,U)>
  Optional(const U& value) :
      value(value),
      hasValue(true) {
    //
  }

  /**
   * Value move constructor. A template is used to ensure that only a value
   * of type T can be implicitly converted to a value of Optional<T>.
   * Implicit type conversions, especially numerical conversions, otherwise
   * cause troublesome ambiguous functions calls when passing a value
   * argument to an optional parameter.
   */
  template<class U, IS_SAME(T,U)>
  Optional(U&& value) :
      value(std::move(value)),
      hasValue(true) {
    //
  }

  /**
   * Conversion constructor. This ensures that if a value of type U can be
   * implicitly converted to type T, then Optional<U> can also be converted
   * to Optional<T>.
   */
  template<class U, IS_CONVERTIBLE(U,T)>
  Optional(const Optional<U>& o) :
      value(o.value),
      hasValue(o.hasValue) {
    //
  }

  /**
   * Conversion constructor. This ensures that if a value of type U can be
   * implicitly converted to type T, then Optional<U> can also be converted
   * to Optional<T>.
   */
  template<class U, IS_CONVERTIBLE(U,T)>
  Optional(Optional<U>&& o) :
      value(std::move(o.value)),
      hasValue(o.hasValue) {
    //
  }

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
 * Optional for array types. Adds additional functionality for implicit
 * conversion of Eigen types.
 *
 * @ingroup libbirch
 *
 * @tparam T Array type.
 */
template<class T>
class Optional<T,std::enable_if_t<is_array<T>::value>> {
  template<class U, class Enable1> friend class Optional;

  static_assert(!std::is_lvalue_reference<T>::value,
      "Optional does not support lvalue reference types.");
public:
  /**
   * Constructor.
   */
  Optional(const Nil& = nil) :
      value(),
      hasValue(false) {
    //
  }

  /**
   * Value copy constructor. A template is used to ensure that only a value
   * of type T can be implicitly converted to a value of Optional<T>.
   * Implicit type conversions, especially numerical conversions, otherwise
   * cause troublesome ambiguous functions calls when passing a value
   * argument to an optional parameter.
   */
  template<class U, IS_SAME(T,U)>
  Optional(const U& value) :
      value(value),
      hasValue(true) {
    //
  }

  /**
   * Value copy constructor for arrays. This allows an argument of an Eigen
   * type to be passed to a parameter of an optional array type.
   */
  template<class EigenType, std::enable_if_t<is_eigen_compatible<T,EigenType>::value,int> = 0>
  Optional(const Eigen::MatrixBase<EigenType>& value) :
      value(value),
      hasValue(true) {
    //
  }

  /**
   * Value move constructor. A template is used to ensure that only a value
   * of type T can be implicitly converted to a value of Optional<T>.
   * Implicit type conversions, especially numerical conversions, otherwise
   * cause troublesome ambiguous functions calls when passing a value
   * argument to an optional parameter.
   */
  template<class U, IS_SAME(T,U)>
  Optional(U&& value) :
      value(std::move(value)),
      hasValue(true) {
    //
  }

  /**
   * Conversion constructor. This ensures that if a value of type U can be
   * implicitly converted to type T, then Optional<U> can also be converted
   * to Optional<T>.
   */
  template<class U, IS_CONVERTIBLE(U,T)>
  Optional(const Optional<U>& o) :
      value(o.value),
      hasValue(o.hasValue) {
    //
  }

  /**
   * Conversion constructor. This ensures that if a value of type U can be
   * implicitly converted to type T, then Optional<U> can also be converted
   * to Optional<T>.
   */
  template<class U, IS_CONVERTIBLE(U,T)>
  Optional(Optional<U>&& o) :
      value(std::move(o.value)),
      hasValue(o.hasValue) {
    //
  }

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
 * Optional for pointer types. Uses a null pointer, rather than a flag, to
 * indicate no value.
 *
 * @ingroup libbirch
 *
 * @tparam T Pointer type.
 */
template<class T>
class Optional<T,std::enable_if_t<is_pointer<T>::value>> {
  template<class U, class Enable1> friend class Optional;

  static_assert(!std::is_lvalue_reference<T>::value,
      "Optional does not support lvalue reference types.");
public:
  /**
   * Constructor.
   */
  Optional(const Nil& = nil) :
      value(nullptr) {
    //
  }

  /**
   * Value copy constructor.
   */
  template<class U, std::enable_if_t<is_base_of<T,U>::value,int> = 0>
  Optional(const U& value) :
      value(value) {
    //
  }

  /**
   * Value move constructor.
   */
  template<class U, std::enable_if_t<is_base_of<T,U>::value,int> = 0>
  Optional(U&& value) :
      value(std::move(value)) {
    //
  }

  /**
   * Copy constructor.
   */
  template<class U, std::enable_if_t<is_base_of<T,U>::value,int> = 0>
  Optional(const Optional<U>& o) :
      value(o.value) {
    //
  }

  /**
   * Move constructor.
   */
  template<class U, std::enable_if_t<is_base_of<T,U>::value,int> = 0>
  Optional(Optional<U>&& o) :
      value(std::move(o.value)) {
    //
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
  T& get() {
    libbirch_assert_msg_(query(), "optional has no value");
    return value;
  }

  /**
   * Get the value.
   */
  const T& get() const {
    libbirch_assert_msg_(query(), "optional has no value");
    return value;
  }

private:
  /**
   * The pointer.
   */
  T value;
};

template<class T>
struct is_value<Optional<T>> {
  static const bool value = is_value<T>::value;
};

template<class T>
struct is_value<Optional<T>&> {
  static const bool value = is_value<T>::value;
};
}
