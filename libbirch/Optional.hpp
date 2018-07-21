/**
 * @file
 */
#pragma once

#include "libbirch/Nil.hpp"

#include "boost/optional.hpp"

namespace bi {
/**
 * Optional.
 *
 * @ingroup libbirch
 *
 * @tparam T Type.
 *
 * @internal While boost::optional might be preferable, it is significantly
 * more complex and handles cases outside those encountered in Birch. It also
 * causes some compile problems when a Birch class uses optionals of its
 * own type in function signatures.
 */
template<class T>
class Optional {
public:
  /**
   * Constructor for no value.
   */
  Optional(const std::nullptr_t& = nullptr) :
      value(),
      hasValue(false) {
    //
  }

  /**
   * Null constructor.
   */
  Optional(const Nil&) :
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
   * Constructor from Boost optional.
   */
  Optional(const boost::optional<T>& o) :
      hasValue(o) {
    if (hasValue) {
      value = o.get();
    }
  }
  Optional(const boost::optional<T&>& o) :
      hasValue(o) {
    if (hasValue) {
      value = o.get();
    }
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
    bi_assert_msg(hasValue, "optional has no value");
    return value;
  }

  /**
   * Get the value.
   */
  const T& get() const {
    bi_assert_msg(hasValue, "optional has no value");
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
class Optional<SharedCOW<T>> {
  template<class U> friend class Optional;
public:
  /**
   * Value constructor.
   */
  Optional(T* value = nullptr) :
      value(value) {
    //
  }

  /**
   * Value constructor.
   */
  Optional(const Nil&) :
      value(nullptr) {
    //
  }

  /**
   * Value constructor.
   */
  Optional(const SharedCOW<T>& value) :
      value(value) {
    //
  }

  /**
   * Value constructor.
   */
  Optional(const WeakCOW<T>& value) :
      value(value) {
    //
  }

  /**
   * Copy constructor.
   */
  template<class U>
  Optional(const Optional<SharedCOW<U>>& o) :
      value(o.value) {
    //
  }

  /**
   * Copy constructor.
   */
  template<class U>
  Optional(const Optional<WeakCOW<U>>& o) :
      value(o.value) {
    //
  }

  /**
   * Value conversion.
   */
  operator WeakCOW<T>() {
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
  SharedCOW<T>& get() {
    bi_assert_msg(query(), "optional has no value");
    return value;
  }

  /**
   * Get the value.
   */
  const SharedCOW<T>& get() const {
    bi_assert_msg(query(), "optional has no value");
    return value;
  }

private:
  /**
   * The contained value, if any.
   */
  SharedCOW<T> value;
};
}
