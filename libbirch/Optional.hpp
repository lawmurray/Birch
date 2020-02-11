/**
 * @file
 */
#pragma once

#include "libbirch/type.hpp"
#include "libbirch/Nil.hpp"

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
   * Constructor.
   */
  template<IS_NOT_VALUE(T)>
  Optional(Label* context, const Nil& = nil) :
      value(),
      hasValue(false) {
    //
  }

  /**
   * Value copy constructor.
   */
  template<IS_NOT_VALUE(T)>
  Optional(Label* context, const T& value) :
      value(context, value),
      hasValue(true) {
    //
  }

  /**
   * Value move constructor.
   */
  template<IS_NOT_VALUE(T)>
  Optional(Label* context, T&& value) :
      value(context, std::move(value)),
      hasValue(true) {
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
   * Copy constructor.
   */
  Optional(const Optional<T>& o) :
      value(o.value),
      hasValue(o.hasValue) {
    //
  }

  /**
   * Copy constructor.
   */
  template<IS_NOT_VALUE(T)>
  Optional(Label* context, const Optional<T>& o) :
      value(context, o.value),
      hasValue(o.hasValue) {
    //
  }

  /**
   * Move constructor.
   */
  Optional(Optional<T>&& o) :
      value(std::move(o.value)),
      hasValue(o.hasValue) {
    //
  }

  /**
   * Move constructor.
   */
  template<IS_NOT_VALUE(T)>
  Optional(Label* context, Optional<T>&& o) :
      value(context, std::move(o.value)),
      hasValue(o.hasValue) {
    //
  }

  /**
   * Deep copy constructor.
   */
  template<IS_NOT_VALUE(T)>
  Optional(Label* context, Label* label, const Optional<T>& o) :
      value(context, label, o.value),
      hasValue(o.hasValue) {
    //
  }

  /**
   * Nil assignment operator.
   */
  Optional& operator=(const Nil& nil) {
    return assign(nil);
  }

  /**
   * Copy assignment operator.
   */
  Optional& operator=(const Optional<T>& o) {
    return assign(o);
  }

  /**
   * Copy assignment operator.
   */
  template<IS_VALUE(T)>
  Optional& operator=(const T& value) {
    return assign(value);
  }

  /**
   * Move assignment operator.
   */
  Optional& operator=(Optional<T>&& o) {
    return assign(std::move(o));
  }

  /**
   * Move assignment operator.
   */
  template<IS_VALUE(T)>
  Optional& operator=(T&& value) {
    return assign(std::move(value));
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

  /**
   * Copy assignment.
   */
  template<IS_VALUE(T)>
  Optional& assign(const Optional<T>& o) {
    value = o.value;
    hasValue = o.hasValue;
    return *this;
  }

  /**
   * Copy assignment.
   */
  template<IS_NOT_VALUE(T)>
  Optional& assign(Label* context, const Optional<T>& o) {
    value.assign(context, o.value);
    hasValue = o.hasValue;
    return *this;
  }

  /**
   * Move assignment.
   */
  template<IS_VALUE(T)>
  Optional& assign(Optional<T>&& o) {
    value = std::move(o.value);
    hasValue = o.hasValue;
    return *this;
  }

  /**
   * Move assignment.
   */
  template<IS_NOT_VALUE(T)>
  Optional& assign(Label* context, Optional<T>&& o) {
    value.assign(context, std::move(o.value));
    hasValue = o.hasValue;
    return *this;
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
class Optional<T,IS_ARRAY(T)> {
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
   * Constructor.
   */
  template<IS_NOT_VALUE(T)>
  Optional(Label* context, const Nil& = nil) :
      value(),
      hasValue(false) {
    //
  }

  /**
   * Value copy constructor.
   */
  template<IS_NOT_VALUE(T)>
  Optional(Label* context, const T& value) :
      value(context, value),
      hasValue(true) {
    //
  }

  /**
   * Value move constructor.
   */
  template<IS_NOT_VALUE(T)>
  Optional(Label* context, T&& value) :
      value(context, std::move(value)),
      hasValue(true) {
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
   * Copy constructor.
   */
  Optional(const Optional<T>& o) :
      value(o.value),
      hasValue(o.hasValue) {
    //
  }

  /**
   * Copy constructor.
   */
  template<IS_NOT_VALUE(T)>
  Optional(Label* context, const Optional<T>& o) :
      value(context, o.value),
      hasValue(o.hasValue) {
    //
  }

  /**
   * Move constructor.
   */
  Optional(Optional<T>&& o) :
      value(std::move(o.value)),
      hasValue(o.hasValue) {
    //
  }

  /**
   * Move constructor.
   */
  template<IS_NOT_VALUE(T)>
  Optional(Label* context, Optional<T>&& o) :
      value(context, std::move(o.value)),
      hasValue(o.hasValue) {
    //
  }

  /**
   * Deep copy constructor.
   */
  template<IS_NOT_VALUE(T)>
  Optional(Label* context, Label* label, const Optional<T>& o) :
      value(context, label, o.value),
      hasValue(o.hasValue) {
    //
  }

  /**
   * Nil assignment operator.
   */
  Optional& operator=(const Nil& nil) {
    return assign(nil);
  }

  /**
   * Copy assignment operator.
   */
  Optional& operator=(const Optional<T>& o) {
    return assign(o);
  }

  /**
   * Copy assignment operator.
   */
  template<IS_VALUE(T)>
  Optional& operator=(const T& value) {
    return assign(value);
  }

  /**
   * Move assignment operator.
   */
  Optional& operator=(Optional<T>&& o) {
    return assign(std::move(o));
  }

  /**
   * Move assignment operator.
   */
  template<IS_VALUE(T)>
  Optional& operator=(T&& value) {
    return assign(std::move(value));
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

  /**
   * Copy assignment.
   */
  template<IS_VALUE(T)>
  Optional& assign(const Optional<T>& o) {
    value = o.value;
    hasValue = o.hasValue;
    return *this;
  }

  /**
   * Copy assignment.
   */
  template<IS_NOT_VALUE(T)>
  Optional& assign(Label* context, const Optional<T>& o) {
    value.assign(context, o.value);
    hasValue = o.hasValue;
    return *this;
  }

  /**
   * Move assignment.
   */
  template<IS_VALUE(T)>
  Optional& assign(Optional<T>&& o) {
    value = std::move(o.value);
    hasValue = o.hasValue;
    return *this;
  }

  /**
   * Move assignment.
   */
  template<IS_NOT_VALUE(T)>
  Optional& assign(Label* context, Optional<T>&& o) {
    value.assign(context, std::move(o.value));
    hasValue = o.hasValue;
    return *this;
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
class Optional<T,IS_POINTER(T)> {
  template<class U, class Enable1> friend class Optional;

  static_assert(!std::is_lvalue_reference<T>::value,
      "Optional does not support lvalue reference types.");
public:
  Optional& operator=(const Optional&) = delete;
  Optional& operator=(Optional&&) = delete;

  /**
   * Constructor.
   */
  Optional(const Nil& = nil) :
      value() {
    //
  }

  /**
   * Constructor.
   */
  Optional(Label* context, typename T::value_type* ptr,
      const bool cross = false) : value(context, ptr, cross) {
    //
  }

  /**
   * Constructor.
   */
  Optional(Label* context, const Nil& = nil) :
      value() {
    //
  }

  /**
   * Value copy constructor.
   */
  Optional(const T& value) :
      value(value) {
    //
  }

  /**
   * Value copy constructor.
   */
  Optional(Label* context, const T& value) :
      value(context, value) {
    //
  }

  /**
   * Value conversion constructor. This ensures that any value of type U that
   * can be implicitly converted to type T can also be convert to type
   * Optional<T>.
   */
  template<class U, IS_CONVERTIBLE(U,T)>
  Optional(const U& value) :
      value(value) {
    //
  }

  /**
   * Value move constructor.
   */
  Optional(T&& value) :
      value(std::move(value)) {
    //
  }

  /**
   * Value move constructor.
   *
   * @todo Make generic while avoiding use of universal reference.
   */
  Optional(Label* context, T&& value) :
      value(context, std::move(value)) {
    //
  }

  /**
   * Copy constructor.
   */
  Optional(const Optional<T>& o) :
      value(o.value) {
    //
  }

  /**
   * Copy constructor.
   */
  Optional(Label* context, const Optional<T>& o) :
      value(context, o.value) {
    //
  }

  /**
   * Conversion constructor. This ensures that if a value of type U can be
   * implicitly converted to type T, then Optional<U> can also be converted
   * to Optional<T>.
   */
  template<class U, IS_CONVERTIBLE(U,T)>
  Optional(const Optional<U>& o) :
      value(o.value) {
    //
  }

  /**
   * Move constructor.
   */
  Optional(Optional<T>&& o) :
      value(std::move(o.value)) {
    //
  }

  /**
   * Move constructor.
   */
  Optional(Label* context, Optional<T>&& o) :
      value(context, std::move(o.value)) {
    //
  }

  /**
   * Deep copy constructor.
   */
  Optional(Label* context, Label* label, const Optional& o) :
      value(context, label, o.value) {
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

  /**
   * Copy assignment.
   */
  Optional& assign(Label* context, const Optional<T>& o) {
    this->value.assign(context, o.value);
    return *this;
  }

  /**
   * Move assignment.
   */
  Optional& assign(Label* context, Optional<T>&& o) {
    this->value.assign(context, std::move(o.value));
    return *this;
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
