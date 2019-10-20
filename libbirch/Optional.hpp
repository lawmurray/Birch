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
template<class T>
class Optional {
  template<class U> friend class Optional;
public:
  /**
   * Constructor.
   */
  Optional() :
      value(),
      hasValue(false) {
    //
  }

  /**
   * Constructor.
   */
  Optional(const Nil&) :
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
   * Constructor.
   */
  Optional(const T& value) :
      value(value),
      hasValue(true) {
    //
  }

  /**
   * Constructor.
   */
  template<class U, typename = std::enable_if_t<std::is_convertible<U,T>::value>>
  Optional(const U& value) :
      value(value),
      hasValue(true) {
    //
  }

  /**
   * Constructor.
   */
  template<IS_NOT_VALUE(T), class U, typename = std::enable_if_t<std::is_convertible<U,T>::value>>
  Optional(Label* context, const U& value) :
      value(context, value),
      hasValue(true) {
    //
  }

  /**
   * Constructor.
   */
  Optional(T&& value) :
      value(std::move(value)),
      hasValue(true) {
    //
  }

  /**
   * Constructor.
   */
  template<class U, typename = std::enable_if_t<std::is_convertible<U,T>::value>>
  Optional(U&& value) :
      value(std::move(value)),
      hasValue(true) {
    //
  }

  /**
   * Constructor.
   */
  template<IS_NOT_VALUE(T), class U, typename = std::enable_if_t<std::is_convertible<U,T>::value>>
  Optional(Label* context, U&& value) :
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
  template<class U, typename = std::enable_if_t<std::is_convertible<U,T>::value>>
  Optional(const Optional<U>& o) :
      value(o.value),
      hasValue(o.hasValue) {
    //
  }

  /**
   * Copy constructor.
   */
  template<IS_NOT_VALUE(T), class U, typename = std::enable_if_t<std::is_convertible<U,T>::value>>
  Optional(Label* context, const Optional<U>& o) :
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
  template<class U, typename = std::enable_if_t<std::is_convertible<U,T>::value>>
  Optional(Optional<U>&& o) :
      value(std::move(o.value)),
      hasValue(o.hasValue) {
    //
  }

  /**
   * Move constructor.
   */
  template<IS_NOT_VALUE(T), class U, typename = std::enable_if_t<std::is_convertible<U,T>::value>>
  Optional(Label* context, Optional<U>&& o) :
      value(context, std::move(o.value)),
      hasValue(o.hasValue) {
    //
  }

  /**
   * Deep copy constructor.
   */
  template<IS_NOT_VALUE(T)>
  Optional(Label* context, Label* label, const Optional& o) :
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
    this->value = o.value;
    this->hasValue = o.hasValue;
    return *this;
  }

  /**
   * Copy assignment.
   */
  template<IS_NOT_VALUE(T)>
  Optional& assign(Label* context, const Optional<T>& o) {
    this->value.assign(context, o.value);
    this->hasValue = o.hasValue;
    return *this;
  }

  /**
   * Move assignment.
   */
  template<IS_VALUE(T)>
  Optional& assign(Optional<T>&& o) {
    this->value = std::move(o.value);
    this->hasValue = o.hasValue;
    return *this;
  }

  /**
   * Move assignment.
   */
  template<IS_NOT_VALUE(T)>
  Optional& assign(Label* context, Optional<T>&& o) {
    this->value.assign(context, std::move(o.value));
    this->hasValue = o.hasValue;
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

template<class T>
struct is_value<Optional<T>> {
  static const bool value = is_value<T>::value;
};

template<class T>
struct is_value<Optional<T>&> {
  static const bool value = is_value<T>::value;
};
}
