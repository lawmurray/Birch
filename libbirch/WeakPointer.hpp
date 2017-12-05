/**
 * @file
 */
#pragma once

#include "libbirch/global.hpp"

namespace bi {
class Allocation;
class Any;
template<class T> class SharedPointer;

/**
 * Weak pointer with copy-on-write semantics.
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
   * Constructor from raw pointer.
   */
  WeakPointer(T* raw = nullptr);

  /**
   * Construct with existing allocation.
   */
  WeakPointer(Allocation* allocation);

  /**
   * Constructor from shared pointer.
   */
  WeakPointer(const SharedPointer<T>& o);

  /**
   * Constructor from weak pointer.
   */
  WeakPointer(const WeakPointer<T>& o) = default;

  /**
   * Assignment from null pointer.
   */
  WeakPointer<T>& operator=(const std::nullptr_t& o);

  /**
   * Assignment from shared pointer.
   */
  WeakPointer<T>& operator=(const SharedPointer<T>& o);

  /**
   * Assignment from weak pointer.
   */
  WeakPointer<T>& operator=(const WeakPointer<T>& o) = default;

  /**
   * Generic assignment from shared pointer.
   */
  template<class U, typename = std::enable_if<std::is_base_of<T,U>::value>>
  WeakPointer<T>& operator=(const SharedPointer<U>& o);

  /**
   * Generic assignment from weak pointer.
   */
  template<class U, typename = std::enable_if<std::is_base_of<T,U>::value>>
  WeakPointer<T>& operator=(const WeakPointer<U>& o);

  /**
   * Value assignment.
   */
  template<class U>
  WeakPointer<T>& operator=(const U& o);

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
  T* get() const;

  /**
   * Dynamic cast. Returns `nullptr` if the cast if unsuccessful.
   */
  template<class U>
  WeakPointer<U> dynamic_pointer_cast() const;

  /**
   * Static cast. Undefined if unsuccessful.
   */
  template<class U>
  WeakPointer<U> static_pointer_cast() const;

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

  /**
   * Comparison operators.
   */
  template<class U>
  bool operator!=(const WeakPointer<U>& o) const {
    return this->raw != o.raw || this->gen != o.gen;
  }
  template<class U>
  bool operator==(const WeakPointer<U>& o) const {
    return this->raw == o.raw && this->gen == o.gen;
  }
};

template<>
class WeakPointer<Any> {
  friend class SharedPointer<Any> ;
public:
  WeakPointer(Any* raw = nullptr);
  WeakPointer(Allocation* allocation);
  WeakPointer(const SharedPointer<Any>& o);
  WeakPointer(const WeakPointer<Any>& o);
  WeakPointer<Any>& operator=(const std::nullptr_t& o);
  WeakPointer<Any>& operator=(const SharedPointer<Any>& o);
  WeakPointer<Any>& operator=(const WeakPointer<Any>& o);
  ~WeakPointer();

  /**
   * Is the pointer not null?
   */
  bool query() const;

  /**
   * Get the pointer.
   */
  Any* get() const;

  /**
   * Release the pointer.
   */
  void release();

  template<class U>
  WeakPointer<U> dynamic_pointer_cast() const;

  template<class U>
  WeakPointer<U> static_pointer_cast() const;

protected:
  /**
   * Allocation control structure.
   */
  Allocation* allocation;
};
}

#include <cassert>

template<class T>
bi::WeakPointer<T>::WeakPointer(T* raw) :
    super_type(raw) {
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
template<class U, typename >
bi::WeakPointer<T>& bi::WeakPointer<T>::operator=(const SharedPointer<U>& o) {
  WeakPointer<Any>::operator=(o);
  return *this;
}

template<class T>
template<class U, typename >
bi::WeakPointer<T>& bi::WeakPointer<T>::operator=(const WeakPointer<U>& o) {
  WeakPointer<Any>::operator=(o);
  return *this;
}

template<class T>
template<class U>
bi::WeakPointer<T>& bi::WeakPointer<T>::operator=(const U& o) {
  *get() = o;
  return *this;
}

template<class T>
template<class U, typename >
bi::WeakPointer<T>::operator U() const {
  return static_cast<U>(*get());
}

template<class T>
T* bi::WeakPointer<T>::get() const {
#ifdef NDEBUG
  return static_cast<T*>(WeakPointer<Any>::get());
#else
  auto raw = WeakPointer<Any>::get();
  if (raw) {
    auto result = dynamic_cast<T*>(raw);
    assert(result);
    return result;
  } else {
    return nullptr;
  }
#endif
}

template<class T>
template<class U>
bi::WeakPointer<U> bi::WeakPointer<T>::dynamic_pointer_cast() const {
  if (dynamic_cast<U*>(get())) {
    return WeakPointer<U>(this->allocation);
  } else {
    return WeakPointer<U>();
  }
}

template<class T>
template<class U>
bi::WeakPointer<U> bi::WeakPointer<T>::static_pointer_cast() const {
#ifndef NDEBUG
  assert(dynamic_cast<U*>(get()));
#endif
  return WeakPointer<U>(this->allocation);
}

template<class T>
bi::WeakPointer<T>::WeakPointer(Allocation* allocation) :
    super_type(allocation) {
  //
}

template<class U>
bi::WeakPointer<U> bi::WeakPointer<bi::Any>::dynamic_pointer_cast() const {
  if (dynamic_cast<U*>(get())) {
    return WeakPointer<U>(this->allocation);
  } else {
    return WeakPointer<U>();
  }
}

template<class U>
bi::WeakPointer<U> bi::WeakPointer<bi::Any>::static_pointer_cast() const {
#ifndef NDEBUG
  assert(dynamic_cast<U*>(get()));
#endif
  return WeakPointer<U>(this->allocation);
}
