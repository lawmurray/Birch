/**
 * @file
 */
#pragma once

#include "libbirch/Nil.hpp"
#include "libbirch/Shared.hpp"
#include "libbirch/Weak.hpp"

namespace libbirch {
/**
 * Optional.
 *
 * @ingroup libbirch
 *
 * @tparam T Type.
 */
template<class T>
class Optional {
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

  /**
   * Copy constructor.
   */
  Optional(const Optional<T>& o) = default;

  /**
   * Move constructor.
   */
  Optional(Optional<T> && o) = default;

  /**
   * Copy assignment.
   */
  Optional<T>& operator=(const Optional<T>& o) = default;

  /**
   * Move assignment.
   */
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
 * Optional for shared pointers. Uses the pointer itself, set to nullptr, to
 * denote a missing value, rather than keeping a separate boolean flag.
 *
 * @ingroup libbirch
 *
 * @tparam T Type.
 */
template<class T>
class Optional<Shared<T>> {
  template<class U> friend class Optional;
public:
  /**
   * Default constructor.
   */
  Optional() = default;

  /**
   * Constructor for no value.
   */
  Optional(const Nil&) :
      value() {
    //
  }

  /**
   * Constructor for value.
   */
  Optional(T* value) :
      value(value) {
    //
  }

  /**
   * Constructor for value.
   */
  template<class U>
  Optional(const Shared<U>& value) :
      value(value) {
    //
  }

  /**
   * Constructor for value.
   */
  template<class U>
  Optional(const Weak<U>& value) :
      value(value) {
    //
  }

  /**
   * Constructor for value.
   */
  template<class U>
  Optional(Shared<U>&& value) :
      value(std::move(value)) {
    //
  }

  /**
   * Constructor for value.
   */
  template<class U>
  Optional(Weak<U>&& value) :
      value(std::move(value)) {
    //
  }

  /**
   * Generic copy constructor.
   */
  template<class U>
  Optional(const Optional<Shared<U>>& o) :
      value(o.value) {
    //
  }

  /**
   * Generic move constructor.
   */
  template<class U>
  Optional(Optional<Shared<U>> && o) :
      value(std::move(o.value)) {
    //
  }

  /**
   * Copy constructor.
   */
  Optional(const Optional<Shared<T>>& o) = default;

  /**
   * Move constructor.
   */
  Optional(Optional<Shared<T>> && o) = default;

  /**
   * Copy assignment.
   */
  Optional<Shared<T>>& operator=(const Optional<Shared<T>>& o) = default;

  /**
   * Move assignment.
   */
  Optional<Shared<T>>& operator=(Optional<Shared<T>> && o) = default;

  /**
   * Value conversion.
   */
  operator Weak<T>() {
    return value;
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
  Shared<T>& get() {
    libbirch_assert_msg_(query(), "optional has no value");
    return value;
  }

  /**
   * Get the value.
   */
  const Shared<T>& get() const {
    libbirch_assert_msg_(query(), "optional has no value");
    return value;
  }

private:
  /**
   * The value, if any.
   */
  Shared<T> value;
};
}
