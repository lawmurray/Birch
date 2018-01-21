/**
 * @file
 */
#pragma once

#include <cassert>

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
   * Constructor for a value.
   */
  Optional(const T& value) :
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
    assert(hasValue);
    return value;
  }
  const T& get() const {
    assert(hasValue);
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
class Optional<SharedPointer<T>> {
public:
  /**
   * Null constructor.
   */
  Optional(const std::nullptr_t& value = nullptr) :
      value(value) {
    //
  }

  /**
   * Value constructor.
   */
  Optional(const SharedPointer<T>& value) :
      value(value) {
    //
  }

  /**
   * Generic value constructor.
   */
  template<class U>
  Optional(const SharedPointer<U>& value) :
      value(value) {
    //
  }

  /**
   * Generic value constructor.
   */
  template<class U>
  Optional(const WeakPointer<U>& value) :
      value(value.lock()) {
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
  SharedPointer<T>& get() {
    assert(query());
    return value;
  }
  const SharedPointer<T>& get() const {
    assert(query());
    return value;
  }

private:
  /**
   * The contained value, if any.
   */
  SharedPointer<T> value;
};
}
