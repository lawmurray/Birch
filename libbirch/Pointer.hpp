/**
 * @file
 */
#pragma once

#include "libbirch/global.hpp"

namespace bi {
class Allocation;

/**
 * Smart pointer with copy-on-write semantics.
 *
 * @ingroup libbirch
 *
 * @tparam T Type.
 */
template<class T>
class Pointer: public Pointer<typename super_type<T>::type> {
  friend class Any;
public:
  typedef T value_type;
  typedef Pointer<T> this_type;
  typedef Pointer<typename super_type<T>::type> super_type;

  /**
   * Constructor.
   */
  Pointer(T* raw = nullptr);

  /**
   * Generic assignment operator.
   */
  template<class U, typename = std::enable_if<std::is_base_of<T,U>::value>>
  Pointer<T>& operator=(const Pointer<U>& o);

  /**
   * Value assignment operator.
   */
  template<class U>
  Pointer<T>& operator=(const U& o);

  /**
   * Null assignment operator.
   */
  Pointer<T>& operator=(const std::nullptr_t& o);

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
   * Cast the pointer. Returns a null pointer if the cast is unsuccessful.
   */
  template<class U>
  Pointer<U> cast() const;

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
  bool operator!=(const Pointer<U>& o) const {
    return this->raw != o.raw || this->gen != o.gen;
  }
  template<class U>
  bool operator==(const Pointer<U>& o) const {
    return this->raw == o.raw && this->gen == o.gen;
  }
};

template<>
class Pointer<Any> {
public:
  Pointer(Any* raw = nullptr);
  Pointer(const Pointer<Any>& o);
  Pointer(Pointer<Any>&& o) = default;
  ~Pointer();
  Pointer<Any>& operator=(const Pointer<Any>& o);
  Pointer<Any>& operator=(Pointer<Any>&& o) = default;
  Pointer<Any>& operator=(const std::nullptr_t& o);

  bool isNull() const;
  Any* get() const;

protected:
  /**
   * Release this pointer. Decrements the shared count and may free the
   * allocation.
   */
  void release();

  /**
   * Allocation control structure.
   */
  Allocation* allocation;
};
}

template<class T>
bi::Pointer<T>::Pointer(T* raw) :
    super_type(raw) {
  //
}

template<class T>
template<class U, typename >
bi::Pointer<T>& bi::Pointer<T>::operator=(const Pointer<U>& o) {
  Pointer<Any>::operator=(o);
  return *this;
}

template<class T>
template<class U>
bi::Pointer<T>& bi::Pointer<T>::operator=(const U& o) {
  *get() = o;
  return *this;
}

template<class T>
bi::Pointer<T>& bi::Pointer<T>::operator=(const std::nullptr_t& o) {
  Pointer<Any>::operator=(o);
  return *this;
}

template<class T>
template<class U, typename >
bi::Pointer<T>::operator U() const {
  return static_cast<U>(*get());
}

template<class T>
T* bi::Pointer<T>::get() const {
#ifdef NDEBUG
  return static_cast<T*>(Pointer<Any>::get());
#else
  auto raw = Pointer<Any>::get();
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
bi::Pointer<U> bi::Pointer<T>::cast() const {
  return Pointer<U>(dynamic_cast<U*>(get()));
}
