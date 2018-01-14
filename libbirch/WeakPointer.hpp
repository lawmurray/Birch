/**
 * @file
 */
#pragma once

#include "libbirch/global.hpp"
#include "libbirch/Optional.hpp"

namespace bi {
/**
 * Shared pointer with copy-on-write semantics.
 *
 * @ingroup libbirch
 *
 * @tparam T Type.
 */
template<class T>
class WeakPointer: public WeakPointer<typename super_type<T>::type> {
public:
  using value_type = T;
  using this_type = WeakPointer<T>;
  using super_type = WeakPointer<typename super_type<T>::type>;
  using root_type = typename super_type::root_type;

  /**
   * Default constructor.
   */
  WeakPointer(const std::nullptr_t& o = nullptr);

  /**
   * Copy constructor.
   */
  WeakPointer(const WeakPointer<T>& o) = default;

  /**
   * Move constructor.
   */
  WeakPointer(WeakPointer<T> && o) = default;

  /**
   * Copy constructor.
   */
  WeakPointer(const SharedPointer<T>& o);

  /**
   * Generic copy constructor.
   */
  template<class U, typename = std::enable_if_t<is_assignable<T,U>::value>>
  WeakPointer(const WeakPointer<U>& o);

  /**
   * Generic copy constructor.
   */
  template<class U, typename = std::enable_if_t<is_assignable<T,U>::value>>
  WeakPointer(const SharedPointer<U>& o);

  /**
   * Assignment from null pointer.
   */
  WeakPointer<T>& operator=(const std::nullptr_t& o);

  /**
   * Copy assignment.
   */
  WeakPointer<T>& operator=(const WeakPointer<T>& o) = default;

  /**
   * Move assignment.
   */
  WeakPointer<T>& operator=(WeakPointer<T> && o) = default;

  /**
   * Assignment from shared pointer.
   */
  WeakPointer<T>& operator=(const SharedPointer<T>& o);

  /**
   * Generic assignment from weak pointer.
   */
  template<class U, typename = std::enable_if<is_assignable<T,U>::value>>
  WeakPointer<T>& operator=(const WeakPointer<U>& o);

  /**
   * Generic assignment from shared pointer.
   */
  template<class U, typename = std::enable_if<is_assignable<T,U>::value>>
  WeakPointer<T>& operator=(const SharedPointer<U>& o);

  /**
   * Value assignment.
   */
  template<class U>
  WeakPointer<T>& operator=(const U& o);

  /**
   * Lock the pointer.
   */
  SharedPointer<T> lock() const;
};

template<>
class WeakPointer<Any> {
  friend class SharedPointer<Any> ;
  friend class WeakPointer<const Any> ;
public:
  using value_type = Any;
  using this_type = WeakPointer<value_type>;
  using root_type = this_type;

  WeakPointer(const std::nullptr_t& o = nullptr);
  WeakPointer(const WeakPointer<Any>& o) = default;
  WeakPointer(WeakPointer<Any> && o) = default;
  WeakPointer(const SharedPointer<Any>& o);
  WeakPointer<Any>& operator=(const std::nullptr_t& o);
  WeakPointer<Any>& operator=(const WeakPointer<Any>& o) = default;
  WeakPointer<Any>& operator=(WeakPointer<Any> && o) = default;
  WeakPointer<Any>& operator=(const SharedPointer<Any>& o);

  SharedPointer<Any> lock() const;

protected:
  /**
   * Wrapped smart pointer.
   */
  std::weak_ptr<Any> ptr;
};

template<>
class WeakPointer<const Any> {
  friend class SharedPointer<const Any> ;
public:
  using value_type = const Any;
  using this_type = WeakPointer<const Any>;
  using root_type = this_type;

  WeakPointer(const std::nullptr_t& o = nullptr);
  WeakPointer(const WeakPointer<const Any>& o) = default;
  WeakPointer(WeakPointer<const Any> && o) = default;
  WeakPointer(const SharedPointer<const Any>& o);
  WeakPointer(const WeakPointer<Any>& o);
  WeakPointer(const SharedPointer<Any>& o);
  WeakPointer<const Any>& operator=(const std::nullptr_t& o);
  WeakPointer<const Any>& operator=(const WeakPointer<const Any>& o) = default;
  WeakPointer<const Any>& operator=(WeakPointer<const Any> && o) = default;
  WeakPointer<const Any>& operator=(const SharedPointer<const Any>& o);
  WeakPointer<const Any>& operator=(const WeakPointer<Any>& o);
  WeakPointer<const Any>& operator=(const SharedPointer<Any>& o);

  SharedPointer<const Any> lock() const;

protected:
  /**
   * Wrapped smart pointer.
   */
  std::weak_ptr<const Any> ptr;
};
}

template<class T>
bi::WeakPointer<T>::WeakPointer(const std::nullptr_t& o) :
    super_type(o) {
  //
}

template<class T>
bi::WeakPointer<T>::WeakPointer(const SharedPointer<T>& o) :
    super_type(o) {
  //
}

template<class T>
template<class U, typename>
bi::WeakPointer<T>::WeakPointer(const WeakPointer<U>& o) :
    super_type(o) {
  //
}

template<class T>
template<class U, typename>
bi::WeakPointer<T>::WeakPointer(const SharedPointer<U>& o) :
    super_type(o) {
  //
}

template<class T>
bi::WeakPointer<T>& bi::WeakPointer<T>::operator=(const std::nullptr_t& o) {
  root_type::operator=(o);
  return *this;
}

template<class T>
bi::WeakPointer<T>& bi::WeakPointer<T>::operator=(const SharedPointer<T>& o) {
  root_type::operator=(o);
  return *this;
}

template<class T>
template<class U, typename >
bi::WeakPointer<T>& bi::WeakPointer<T>::operator=(const SharedPointer<U>& o) {
  root_type::operator=(o);
  return *this;
}

template<class T>
bi::SharedPointer<T> bi::WeakPointer<T>::lock() const {
#ifndef NDEBUG
  return root_type::lock().template dynamic_pointer_cast<T>();
#else
  return root_type::lock().template static_pointer_cast<T>();
#endif
}
