/**
 * @file
 */
#pragma once

namespace bi {
/**
 * Optional.
 *
 * @ingroup library
 *
 * @tparam T Type.
 */
template<class T>
class Optional {
public:
  /**
   * Constructor for no value.
   */
  Optional() : value(), hasValue(false) {
    //
  }

  /**
   * Constructor for a value.
   */
  Optional(const T& value) : value(value), hasValue(true) {
    //
  }

  /**
   * Assign no value.
   */
  Optional<T>& operator=(const std::nullptr_t&) {
    this->value = T();  // to release for garbage collection
    this->hasValue = false;
    return *this;
  }

  /**
   * Assign a value.
   */
  Optional<T>& operator=(const T& value) {
    this->value = value;
    this->hasValue = true;
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
}
