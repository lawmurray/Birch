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
  typedef T value_type;
  typedef WeakPointer<T> this_type;
  typedef WeakPointer<typename super_type<T>::type> super_type;

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
  template<class U, typename = std::enable_if<std::is_base_of<T,U>::value>>
  WeakPointer<T>& operator=(const WeakPointer<U>& o);

  /**
   * Generic assignment from shared pointer.
   */
  template<class U, typename = std::enable_if<std::is_base_of<T,U>::value>>
  WeakPointer<T>& operator=(const SharedPointer<U>& o);

  /**
   * Value assignment.
   */
  template<class U>
  WeakPointer<T>& operator=(const U& o);

  /**
   * Lock the pointer.
   */
  Optional<SharedPointer<T>> lock() const;
};

template<>
class WeakPointer<Any> {
  friend class SharedPointer<Any> ;
public:
  WeakPointer(const std::nullptr_t& o = nullptr);
  WeakPointer(const WeakPointer<Any>& o) = default;
  WeakPointer(WeakPointer<Any> && o) = default;
  WeakPointer(const SharedPointer<Any>& o);
  WeakPointer<Any>& operator=(const std::nullptr_t& o);
  WeakPointer<Any>& operator=(const WeakPointer<Any>& o) = default;
  WeakPointer<Any>& operator=(WeakPointer<Any> && o) = default;
  WeakPointer<Any>& operator=(const SharedPointer<Any>& o);

  Optional<SharedPointer<Any>> lock() const;

protected:
  /**
   * Wrapped smart pointer.
   */
  std::weak_ptr<Any> ptr;
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
bi::WeakPointer<T>& bi::WeakPointer<T>::operator=(const std::nullptr_t& o) {
  WeakPointer<Any>::operator=(o);
  return *this;
}

template<class T>
bi::WeakPointer<T>& bi::WeakPointer<T>::operator=(const SharedPointer<T>& o) {
  WeakPointer<Any>::operator=(o);
  return *this;
}

template<class T>
template<class U, typename >
bi::WeakPointer<T>& bi::WeakPointer<T>::operator=(const SharedPointer<U>& o) {
  WeakPointer<Any>::operator=(o);
  return *this;
}

template<class T>
bi::Optional<bi::SharedPointer<T>> bi::WeakPointer<T>::lock() const {
#ifndef NDEBUG
  return WeakPointer<Any>::lock().template dynamic_pointer_cast<T>();
#else
  return WeakPointer<Any>::lock().template static_pointer_cast<T>();
#endif
}
