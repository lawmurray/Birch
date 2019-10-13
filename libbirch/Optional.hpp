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
 */
template<class T, class Enable = void>
class Optional {
  //
};

/**
 * Optional for pointer types. Uses the pointer itself, set to `nullptr`, to
 * denote a missing value, rather than keeping a separate boolean flag.
 *
 * @ingroup libbirch
 *
 * @tparam T Pointer type.
 */
template<class T>
class Optional<T,IS_POINTER(T)> {
  template<class T1, class Enable1> friend class Optional;
public:
  Optional(const Optional& o) = default;
  Optional(Optional&& o) = default;
  Optional& operator=(const Optional& o) = delete;
  Optional& operator=(Optional&& o) = delete;

  /**
   * Constructor.
   */
  Optional(const Nil& = nil) {
    //
  }

  /**
   * Implicit conversion from value type.
   */
  template<class U>
  Optional(const U& value) :
      value(value) {
    //
  }

  /**
   * Constructor.
   */
  template<class U>
  Optional(Label* context, const U& value) :
      value(context, value) {
    //
  }

  /**
   * Copy constructor.
   */
  template<class U>
  Optional(Label* context, const Optional<U>& o) :
      value(context, o.value) {
    //
  }

  /**
   * Move constructor.
   */
  template<class U>
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
   * Nil assignment.
   */
  Optional& assign(Label* context, const Nil&) {
    this->value.assign(context, Optional<T>());
    return *this;
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

  void freeze() {
    if (value) {
      value.freeze();
    }
  }

  void thaw(Label* label) {
    if (value) {
      value.thaw(label);
    }
  }

  void finish() {
    if (value) {
      value.finish();
    }
  }

private:
  /**
   * The value. The special value `nullptr` denotes no value.
   */
  T value;
};

/**
 * Optional for value types.
 *
 * @ingroup libbirch
 *
 * @tparam T Non-pointer type.
 */
template<class T>
class Optional<T,IS_VALUE(T)> {
  template<class T1, class Enable1> friend class Optional;
public:
  Optional(const Optional&) = default;
  Optional(Optional&&) = default;
  Optional& operator=(const Optional&) = default;
  Optional& operator=(Optional&&) = default;

  /**
   * Constructor.
   */
  Optional(const Nil& = nil) :
      value(),
      hasValue(false) {
    //
  }

  /**
   * Implicit conversion from value type.
   */
  template<class U>
  Optional(const U& value) :
      value(value),
      hasValue(true) {
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

  void freeze() {
    //
  }

  void thaw(Label* label) {
    //
  }

  void finish() {
    //
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
 * Optional for non-pointer and non-value types.
 *
 * @ingroup libbirch
 *
 * @tparam T Non-pointer type.
 */
template<class T>
class Optional<T,IS_NOT_VALUE_NOR_POINTER(T)> {
  template<class T1, class Enable1> friend class Optional;
public:
  Optional(const Optional&) = default;
  Optional(Optional&&) = default;
  Optional& operator=(const Optional&) = delete;
  Optional& operator=(Optional&&) = delete;

  /**
   * Constructor.
   */
  Optional(const Nil& = nil) :
      value(),
      hasValue(false) {
    //
  }

  /**
   * Implicit conversion from value type.
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
  template<class U>
  Optional(Label* context, const U& value) :
      value(context, value),
      hasValue(true) {
    //
  }

  /**
   * Copy constructor.
   */
  template<class U>
  Optional(Label* context, const Optional<U>& o) :
      value(context, o.value),
      hasValue(o.hasValue) {
    //
  }

  /**
   * Move constructor.
   */
  template<class U>
  Optional(Label* context, Optional<U>&& o) :
      value(context, std::move(o.value)),
      hasValue(o.hasValue) {
    //
  }

  /**
   * Deep copy constructor.
   */
  Optional(Label* context, Label* label, const Optional& o) :
      value(context, label, o.value),
      hasValue(o.hasValue) {
    //
  }

  /**
   * Nil assignment.
   */
  Optional& assign(Label* context, const Nil&) {
    this->value.assign(context, Optional<T>());
    this->hasValue = false;
    return *this;
  }

  /**
   * Copy assignment.
   */
  Optional& assign(Label* context, const Optional<T>& o) {
    this->value.assign(context, o.value);
    this->hasValue = o.hasValue;
    return *this;
  }

  /**
   * Move assignment.
   */
  Optional& assign(Label* context, Optional<T>&& o) {
    this->value.assign(context, std::move(o.value));
    this->hasValue = o.hasValue;
    return *this;
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

  void freeze() {
    if (value) {
      value.freeze();
    }
  }

  void thaw(Label* label) {
    if (value) {
      value.thaw(label);
    }
  }

  void finish() {
    if (value) {
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
void freeze(Optional<T>& o) {
  o.freeze();
}

template<class T>
void thaw(Optional<T>& o, LazyLabel* label) {
  o.thaw(label);
}

template<class T>
void finish(Optional<T>& o) {
  o.finish();
}

}
