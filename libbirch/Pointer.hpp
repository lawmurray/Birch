/**
 * @file
 */
#pragma once

#include "libbirch/global.hpp"

namespace bi {
/**
 * Smart pointer for fibers with copy-on-write semantics.
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
   * Raw pointer assignment operator.
   */
  Pointer<T>& operator=(T* raw);

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

  /**
   * Cast the pointer. Returns a null pointer if the case is unsuccessful.
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
  friend struct std::hash<bi::Pointer<bi::Any>>;
  friend struct std::equal_to<bi::Pointer<bi::Any>>;
public:
  Pointer(Any* raw = nullptr);
  Pointer<Any>& operator=(Any* raw);

  bool isNull() const;
  Any* get() const;

  /**
   * The world in which this pointer exists.
   */
  FiberWorld* world;

protected:
  /**
   * Raw pointer.
   */
  Any* raw;
};
}

namespace std {
template<>
struct hash<bi::Pointer<bi::Any>> : public std::hash<int64_t> {
  size_t operator()(const bi::Pointer<bi::Any>& o) const;
};

template<>
struct equal_to<bi::Pointer<bi::Any>> {
  bool operator()(const bi::Pointer<bi::Any>& o1,
      const bi::Pointer<bi::Any>& o2) const;
};
}

template<class T>
bi::Pointer<T>::Pointer(T* raw) :
    super_type(raw) {
  //
}

template<class T>
bi::Pointer<T>& bi::Pointer<T>::operator=(T* raw) {
  Pointer<Any>::operator=(raw);
  return *this;
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
  return Pointer<U>(dynamic_cast<U*>(this->raw), this->gen);
}
