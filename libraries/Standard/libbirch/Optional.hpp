/**
 * @file
 */
#pragma once

#include "libbirch/type.hpp"
#include "libbirch/Nil.hpp"
#include "libbirch/Lazy.hpp"

namespace libbirch {
/**
 * Optional.
 *
 * @ingroup libbirch
 *
 * @tparam T Value type.
 */
template<class T>
class Optional {
  template<class U> friend class Optional;
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
   * Generic value copy constructor.
   *
   * @note SFINAE here prevents implicit numerical conversion, i.e. the type
   * must must be T, not something numerically convertible to T.
   */
  template<class U, std::enable_if_t<std::is_same<U,T>::value,int> = 0>
  Optional(const U& value) :
      value(value),
      hasValue(true) {
    //
  }

  /**
   * Generic value move constructor.
   *
   * @note SFINAE here prevents implicit numerical conversion, i.e. the type
   * must must be T, not something numerically convertible to T.
   */
  template<class U, std::enable_if_t<std::is_same<U,T>::value,int> = 0>
  Optional(U&& value) :
      value(std::move(value)),
      hasValue(true) {
    //
  }

  /**
   * Nil assignment.
   */
  Optional& operator=(const Nil&) {
    this->value = T();
    this->hasValue = false;
    return *this;
  }

  /**
   * Generic assignment.
   */
  template<class U>
  Optional& operator=(const U& value) {
    this->value = value;
    this->hasValue = true;
    return *this;
  }

  /**
   * Accept visitor.
   */
  template<class Visitor>
  void accept_(const Visitor& v) {
    if (hasValue) {
      v.visit(value);
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
 * Optional specialization for pointer types. Uses a null pointer to
 * represent no value, rather than using an additional flag.
 *
 * @ingroup libbirch
 *
 * @tparam T Pointer type.
 */
template<class P>
class Optional<Lazy<P>> {
  template<class U> friend class Optional;
public:
  /**
   * Constructor.
   */
  Optional(const Nil& = nil) :
      value(nullptr) {
    //
  }

  /**
   * Constructor.
   */
  Optional(typename P::value_type* ptr) :
      value(ptr) {
    //
  }

  /**
   * Generic value copy constructor.
   */
  template<class Q, std::enable_if_t<std::is_base_of<typename P::value_type,
      typename Q::value_type>::value,int> = 0>
  Optional(const Lazy<Q>& value) :
      value(value) {
    //
  }

  /**
   * Generic value move constructor.
   */
  template<class Q, std::enable_if_t<std::is_base_of<typename P::value_type,
      typename Q::value_type>::value,int> = 0>
  Optional(Lazy<Q>&& value) :
      value(std::move(value)) {
    //
  }

  /**
   * Generic copy constructor.
   */
  template<class Q, std::enable_if_t<std::is_base_of<typename P::value_type,
      typename Q::value_type>::value,int> = 0>
  Optional(const Optional<Lazy<Q>>& o) :
      value(o.value) {
    //
  }

  /**
   * Generic move constructor.
   */
  template<class Q, std::enable_if_t<std::is_base_of<typename P::value_type,
      typename Q::value_type>::value,int> = 0>
  Optional(Optional<Lazy<Q>>&& o) :
      value(std::move(o.value)) {
    //
  }

  /**
   * Generic value copy assignment.
   */
  template<class Q, std::enable_if_t<std::is_base_of<typename P::value_type,
      typename Q::value_type>::value,int> = 0>
  Optional& operator=(const Lazy<Q>& value) {
    this->value = value;
    return *this;
  }

  /**
   * Generic value move assignment.
   */
  template<class Q, std::enable_if_t<std::is_base_of<typename P::value_type,
      typename Q::value_type>::value,int> = 0>
  Optional& operator=(Lazy<Q>&& value) {
    this->value = std::move(value);
    return *this;
  }

  /**
   * Generic copy assignment.
   */
  template<class Q, std::enable_if_t<std::is_base_of<typename P::value_type,
      typename Q::value_type>::value,int> = 0>
  Optional& operator=(const Optional<Lazy<Q>>& o) {
    value = o.value;
    return *this;
  }

  /**
   * Generic move assignment.
   */
  template<class Q, std::enable_if_t<std::is_base_of<typename P::value_type,
      typename Q::value_type>::value,int> = 0>
  Optional& operator=(Optional<Lazy<Q>>&& o) {
    value = std::move(o.value);
    return *this;
  }

  /**
   * Generic assignment.
   */
  template<class T, std::enable_if_t<std::is_assignable<P,T>::value,int> = 0>
  Optional& operator=(const T& value) {
    this->value = value;
    return *this;
  }

  /**
   * Accept visitor.
   */
  template<class Visitor>
  void accept_(const Visitor& v) {
    if (value.query()) {
      v.visit(value);
    }
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
  Lazy<P>& get() {
    libbirch_assert_msg_(query(), "optional has no value");
    return value;
  }

  /**
   * Get the value.
   */
  const Lazy<P>& get() const {
    libbirch_assert_msg_(query(), "optional has no value");
    return value;
  }

private:
  /**
   * The pointer.
   */
  Lazy<P> value;
};

template<class T>
struct is_value<Optional<T>> {
  static const bool value = is_value<T>::value;
};

template<class T, unsigned N>
struct is_acyclic<Optional<T>,N> {
  static const bool value = is_acyclic<T,N>::value;
};

template<class T>
auto canonical(const Optional<T>& o) {
  return o;
}

template<class T>
auto canonical(Optional<T>&& o) {
  return std::move(o);
}

}

