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
  template<class U, IS_CONVERTIBLE(U,T)>
  Optional(const U& value) :
      value(value),
      hasValue(true) {
    //
  }

  /**
   * Constructor.
   */
  template<IS_NOT_VALUE(T), class U, IS_CONVERTIBLE(U,T)>
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
  template<class U, IS_CONVERTIBLE(U,T)>
  Optional(U&& value) :
      value(std::move(value)),
      hasValue(true) {
    //
  }

  /**
   * Constructor.
   */
  template<IS_NOT_VALUE(T), class U, IS_CONVERTIBLE(U,T)>
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
  template<class U, IS_CONVERTIBLE(U,T)>
  Optional(const Optional<U>& o) :
      value(o.value),
      hasValue(o.hasValue) {
    //
  }

  /**
   * Copy constructor.
   */
  template<IS_NOT_VALUE(T), class U, IS_CONVERTIBLE(U,T)>
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
  template<class U, IS_CONVERTIBLE(U,T)>
  Optional(Optional<U>&& o) :
      value(std::move(o.value)),
      hasValue(o.hasValue) {
    //
  }

  /**
   * Move constructor.
   */
  template<IS_NOT_VALUE(T), class U, IS_CONVERTIBLE(U,T)>
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

/**
 * Optional for pointer types. Uses a null pointer, rather than a flag, to
 * indicate no value.
 *
 * @ingroup libbirch
 *
 * @tparam T Type type.
 */
template<class T>
class Optional<T,std::enable_if_t<is_pointer<T>::value>> {
  template<class U, class Enable1> friend class Optional;
public:
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
  Optional(Label* context, const Nil& = nil) :
      value() {
    //
  }

  /**
   * Constructor.
   */
  Optional(const T& value) :
      value(value) {
    //
  }

  /**
   * Constructor.
   */
  template<class U, IS_CONVERTIBLE(U,T)>
  Optional(const U& value) :
      value(value) {
    //
  }

  /**
   * Constructor.
   */
  template<class U, IS_CONVERTIBLE(U,T)>
  Optional(Label* context, const U& value) :
      value(context, value) {
    //
  }

  /**
   * Constructor.
   */
  Optional(T&& value) :
      value(std::move(value)) {
    //
  }

  /**
   * Constructor.
   */
  template<class U, IS_CONVERTIBLE(U,T)>
  Optional(U&& value) :
      value(std::move(value)) {
    //
  }

  /**
   * Constructor.
   */
  template<class U, IS_CONVERTIBLE(U,T)>
  Optional(Label* context, U&& value) :
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
   * Move assignment operator.
   */
  Optional& operator=(Optional<T>&& o) {
    return assign(std::move(o));
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
