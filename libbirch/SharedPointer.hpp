/**
 * @file
 */
#pragma once

#include "libbirch/global.hpp"

namespace bi {
/**
 * Shared pointer with copy-on-write semantics.
 *
 * @ingroup libbirch
 *
 * @tparam T Type.
 */
template<class T>
class SharedPointer: public SharedPointer<typename super_type<T>::type> {
  friend class WeakPointer<T> ;
public:
  typedef T value_type;
  typedef SharedPointer<T> this_type;
  typedef SharedPointer<typename super_type<T>::type> super_type;

  /**
   * Default constructor.
   */
  SharedPointer();

  /**
   * Null constructor.
   */
  SharedPointer(const std::nullptr_t& o);

  /**
   * Constructor from raw pointer.
   */
  SharedPointer(T* raw);

  /**
   * Constructor for STL shared pointer.
   */
  SharedPointer(const std::shared_ptr<T>& ptr);

  /**
   * Copy constructor.
   */
  SharedPointer(const SharedPointer<T>& o) = default;

  /**
   * Move constructor.
   */
  SharedPointer(SharedPointer<T> && o) = default;

  /**
   * Assignment from null pointer.
   */
  SharedPointer<T>& operator=(const std::nullptr_t& o);

  /**
   * Copy assignment.
   */
  SharedPointer<T>& operator=(const SharedPointer<T>& o) = default;

  /**
   * Move assignment.
   */
  SharedPointer<T>& operator=(SharedPointer<T> && o) = default;

  /**
   * Generic assignment from shared pointer.
   */
  template<class U, typename = std::enable_if<std::is_base_of<T,U>::value>>
  SharedPointer<T>& operator=(const SharedPointer<U>& o);

  /**
   * Value assignment.
   */
  template<class U>
  SharedPointer<T>& operator=(const U& o);

  /**
   * User-defined conversions. This allows pointers to be passed as arguments
   * to functions with value type parameters, where the type of the object
   * pointed to has a conversion to the value type.
   *
   * @seealso has_conversion
   */
  template<class U, typename = std::enable_if_t<has_conversion<T,U>::value>>
  operator U() const;

  /**
   * Get the raw pointer.
   */
  T* get();
  T* get() const;

  /**
   * Dynamic cast. Returns `nullptr` if the cast if unsuccessful.
   */
  template<class U>
  SharedPointer<U> dynamic_pointer_cast() const;

  /**
   * Static cast. Undefined if unsuccessful.
   */
  template<class U>
  SharedPointer<U> static_pointer_cast() const;

  /**
   * Dereference.
   */
  T& operator*() {
    return *get();
  }
  const T& operator*() const {
    return *get();
  }

  /**
   * Member access.
   */
  T* operator->() {
    return get();
  }
  T* const operator->() const {
    return get();
  }

  /**
   * Call operator.
   */
  template<class ... Args>
  auto operator()(Args ... args) {
    return (*get())(args...);
  }
  template<class ... Args>
  auto operator()(Args ... args) const {
    return (*get())(args...);
  }
};

template<>
class SharedPointer<Any> {
  friend class WeakPointer<Any> ;
public:
  typedef Any value_type;

  SharedPointer();
  SharedPointer(const std::nullptr_t& o);
  SharedPointer(Any* raw);
  SharedPointer(const std::shared_ptr<Any>& ptr);
  SharedPointer(const SharedPointer<Any>& o) = default;
  SharedPointer(SharedPointer<Any> && o) = default;
  SharedPointer<Any>& operator=(const std::nullptr_t& o);
  SharedPointer<Any>& operator=(const SharedPointer<Any>& o) = default;
  SharedPointer<Any>& operator=(SharedPointer<Any> && o) = default;

  /**
   * Is the pointer not null?
   */
  bool query() const;

  /**
   * Get the pointer.
   */
  Any* get();
  Any* get() const;

  template<class U>
  SharedPointer<U> dynamic_pointer_cast() const;

  template<class U>
  SharedPointer<U> static_pointer_cast() const;

  /**
   * Dereference.
   */
  Any& operator*() {
    return *get();
  }
  const Any& operator*() const {
    return *get();
  }

  /**
   * Member access.
   */
  Any* operator->() {
    return get();
  }
  Any* const operator->() const {
    return get();
  }

protected:
  /**
   * Wrapped smart pointer.
   */
  std::shared_ptr<Any> ptr;
};
}

template<class T>
bi::SharedPointer<T>::SharedPointer() :
    super_type(new T()) {
  //
}

template<class T>
bi::SharedPointer<T>::SharedPointer(const std::nullptr_t& o) :
    super_type(o) {
  //
}

template<class T>
bi::SharedPointer<T>::SharedPointer(T* raw) :
    super_type(raw) {
  //
}

template<class T>
bi::SharedPointer<T>::SharedPointer(const std::shared_ptr<T>& ptr) :
    super_type(ptr) {
  //
}

template<class T>
bi::SharedPointer<T>& bi::SharedPointer<T>::operator=(
    const std::nullptr_t& o) {
  SharedPointer<Any>::operator=(o);
  return *this;
}

template<class T>
template<class U, typename >
bi::SharedPointer<T>& bi::SharedPointer<T>::operator=(
    const SharedPointer<U>& o) {
  SharedPointer<Any>::operator=(o);
  return *this;
}

template<class T>
template<class U>
bi::SharedPointer<T>& bi::SharedPointer<T>::operator=(const U& o) {
  *get() = o;
  return *this;
}

template<class T>
template<class U, typename >
bi::SharedPointer<T>::operator U() const {
  return static_cast<U>(*get());
}

template<class T>
T* bi::SharedPointer<T>::get() {
#ifndef NDEBUG
  return dynamic_cast<T*>(SharedPointer<Any>::get());
#else
  return static_cast<T*>(SharedPointer<Any>::get());
#endif
}

template<class T>
T* bi::SharedPointer<T>::get() const {
#ifndef NDEBUG
  return dynamic_cast<T*>(SharedPointer<Any>::get());
#else
  return static_cast<T*>(SharedPointer<Any>::get());
#endif
}

template<class T>
template<class U>
bi::SharedPointer<U> bi::SharedPointer<T>::dynamic_pointer_cast() const {
  return SharedPointer<U>(std::dynamic_pointer_cast<U>(this->ptr));
}

template<class T>
template<class U>
bi::SharedPointer<U> bi::SharedPointer<T>::static_pointer_cast() const {
  return SharedPointer<U>(std::static_pointer_cast<U>(this->ptr));
}

template<class U>
bi::SharedPointer<U> bi::SharedPointer<bi::Any>::dynamic_pointer_cast() const {
  return SharedPointer<U>(std::dynamic_pointer_cast<U>(ptr));
}

template<class U>
bi::SharedPointer<U> bi::SharedPointer<bi::Any>::static_pointer_cast() const {
  return SharedPointer<U>(std::static_pointer_cast<U>(ptr));
}
