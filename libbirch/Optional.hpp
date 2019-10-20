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
public:
  /**
   * Nil constructor.
   */
  Optional(const Nil& = nil) :
      value(),
      hasValue(false) {
    //
  }

  /**
   * Nil constructor.
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
  template<class U>
  Optional(const U& value) :
      value(value),
      hasValue(true) {
    //
  }

  /**
   * Constructor.
   */
  template<IS_NOT_VALUE(T)>
  Optional(Label* context, const T& value) :
      value(context, value),
      hasValue(true) {
    //
  }

  /**
   * Constructor.
   */
  template<class U>
  Optional(U&& value) :
      value(std::move(value)),
      hasValue(true) {
    //
  }

  /**
   * Constructor.
   */
  template<IS_NOT_VALUE(T)>
  Optional(Label* context, T&& value) :
      value(context, std::move(value)),
      hasValue(true) {
    //
  }

  /**
   * Copy constructor for value type. Return conversion for all types.
   */
  Optional(const Optional<T>& o) :
      value(o.value),
      hasValue(o.hasValue) {
    //
  }

  /**
   * Copy constructor.
   */
  template<class U>
  Optional(const Optional<U>& o) :
      value(o.value),
      hasValue(o.hasValue) {
    //
  }

  /**
   * Copy constructor.
   */
  template<IS_NOT_VALUE(T), class U>
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
  template<class U>
  Optional(Optional<U>&& o) :
      value(std::move(o.value)),
      hasValue(o.hasValue) {
    //
  }

  /**
   * Move constructor.
   */
  template<IS_NOT_VALUE(T), class U>
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
  template<IS_VALUE(T), class U>
  Optional& operator=(const Optional<U>& o) {
    return assign(o);
  }

  /**
   * Copy assignment operator.
   */
  template<IS_VALUE(T), class U>
  Optional& operator=(const U& value) {
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
  template<IS_VALUE(T), class U>
  Optional& operator=(Optional<U>&& o) {
    return assign(std::move(o));
  }

  /**
   * Move assignment operator.
   */
  template<IS_VALUE(T), class U>
  Optional& operator=(U&& value) {
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
   * Nil assignment.
   */
  template<IS_VALUE(T)>
  Optional& assign(const Nil&) {
    this->value = T();
    this->hasValue = false;
    return *this;
  }

  /**
   * Nil assignment.
   */
  template<IS_NOT_VALUE(T)>
  Optional& assign(Label* context, const Nil&) {
    this->value.assign(context, T());
    this->hasValue = false;
    return *this;
  }

  /**
   * Copy assignment.
   */
  template<IS_VALUE(T), class U>
  Optional& assign(const Optional<U>& o) {
    this->value = o.value;
    this->hasValue = o.hasValue;
    return *this;
  }

  /**
   * Copy assignment.
   */
  template<IS_NOT_VALUE(T), class U>
  Optional& assign(Label* context, const Optional<U>& o) {
    this->value.assign(context, o.value);
    this->hasValue = o.hasValue;
    return *this;
  }

  /**
   * Copy assignment.
   */
  template<IS_VALUE(T), class U>
  Optional& assign(const U& value) {
    this->value = value;
    this->hasValue = true;
    return *this;
  }

  /**
   * Copy assignment.
   */
  template<IS_NOT_VALUE(T), class U>
  Optional& assign(Label* context, const U& value) {
    this->value.assign(context, value);
    this->hasValue = true;
    return *this;
  }

  /**
   * Move assignment.
   */
  template<IS_VALUE(T), class U>
  Optional& assign(Optional<U>&& o) {
    this->value = std::move(o.value);
    this->hasValue = o.hasValue;
    return *this;
  }

  /**
   * Move assignment.
   */
  template<IS_NOT_VALUE(T), class U>
  Optional& assign(Label* context, Optional<U>&& o) {
    this->value.assign(context, std::move(o.value));
    this->hasValue = o.hasValue;
    return *this;
  }

  /**
   * Move assignment.
   */
  template<IS_VALUE(T), class U>
  Optional& assign(U&& value) {
    this->value = std::move(value);
    this->hasValue = true;
    return *this;
  }

  /**
   * Move assignment.
   */
  template<IS_NOT_VALUE(T), class U>
  Optional& assign(Label* context, U&& value) {
    this->value.assign(context, std::move(value));
    this->hasValue = true;
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
