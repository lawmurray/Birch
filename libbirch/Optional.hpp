/**
 * @file
 */
#pragma once

namespace bi {
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
   * Constructor for no value.
   */
  Optional() :
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

/**
 * Optional for pointer types. Uses the pointer itself, set to nullptr, to
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
   * Constructor for no value.
   */
  Optional() :
      value() {
    //
  }

  /**
   * Constructor for a value.
   */
  Optional(const SharedPointer<T>& value) :
      value(value) {
    //
  }

  /**
   * Assign no value.
   */
  Optional<SharedPointer<T>>& operator=(const std::nullptr_t& o) {
    this->value = o;
    return *this;
  }

  /**
   * Assign a value.
   */
  Optional<SharedPointer<T>>& operator=(const SharedPointer<T>& value) {
    this->value = value;
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
