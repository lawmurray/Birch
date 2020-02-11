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
 * @tparam T Type type.
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
  template<IS_VALUE(T)>
  Optional(const T& value) :
      value(value),
      hasValue(true) {
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
  template<IS_VALUE(T)>
  Optional(T&& value) :
      value(std::move(value)),
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

  template<IS_VALUE(T)>
  void freeze() {
    //
  }

  template<IS_NOT_VALUE(T)>
  void freeze() {
    if (hasValue) {
      value.freeze();
    }
  }

  template<IS_VALUE(T)>
  void thaw(Label* label) {
    //
  }

  template<IS_NOT_VALUE(T)>
  void thaw(Label* label) {
    if (hasValue) {
      value.thaw(label);
    }
  }

  template<IS_VALUE(T)>
  void finish() {
    //
  }

  template<IS_NOT_VALUE(T)>
  void finish() {
    if (hasValue) {
      value.finish();
    }
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
 * @tparam T Type type.
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
  Optional(Label* context, typename T::value_type* ptr, const bool cross = false) :
      value(context, ptr, cross) {
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
  template<class U, IS_CONVERTIBLE(U,T)>
  Optional(const U& value) :
      value(value) {
    //
  }

  /**
   * Value copy constructor.
   */
  template<class U, IS_CONVERTIBLE(U,T)>
  Optional(Label* context, const U& value) :
      value(context, value) {
    //
  }

  /**
   * Value move constructor.
   *
   * @todo Make generic while avoiding use of universal reference.
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
  template<class U, IS_CONVERTIBLE(U,T)>
  Optional(const Optional<U>& o) :
      value(o.value) {
    //
  }

  /**
   * Copy constructor.
   */
  template<class U, IS_CONVERTIBLE(U,T)>
  Optional(Label* context, const Optional<U>& o) :
      value(context, o.value) {
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
  template<class U, IS_CONVERTIBLE(U,T)>
  Optional(Optional<U>&& o) :
      value(std::move(o.value)) {
    //
  }

  /**
   * Move constructor.
   */
  template<class U, IS_CONVERTIBLE(U,T)>
  Optional(Label* context, Optional<U>&& o) :
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

  void freeze() {
    if (query()) {
      get().freeze();
    }
  }

  void thaw(Label* label) {
    if (query()) {
      get().thaw(label);
    }
  }

  void finish() {
    if (query()) {
      get().finish();
    }
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
