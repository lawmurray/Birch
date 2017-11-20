/**
 * @file
 */
#pragma once

#include "bi/lib/global.hpp"

#include "boost/optional.hpp"

#include <cstdint>

namespace bi {
/**
 * Smart pointer for fibers with copy-on-write semantics.
 *
 * @ingroup library
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
   * Raw pointer constructor.
   */
  Pointer(T* raw = nullptr);

  /**
   * Copy constructor.
   */
  Pointer(const Pointer<T>& o) = default;

  /**
   * Move constructor.
   */
  Pointer(Pointer<T>&& o) = default;

  /**
   * Copy assignment operator.
   */
  Pointer<T>& operator=(const Pointer<T>& o) = default;

  /**
   * Move assignment operator.
   */
  Pointer<T>& operator=(Pointer<T>&& o) = default;

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
  operator U();
  template<class U, typename = std::enable_if_t<has_conversion<T,U>::value>>
  operator U() const;

  /**
   * Get the raw pointer.
   */
  T* get();
  T* const get() const;

  /**
   * Cast the pointer.
   */
  template<class U>
  boost::optional<Pointer<U>> cast() const;

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
  auto operator()(Args ... args) const {
    return (*get())(args...);
  }
};

template<>
class Pointer<Any> {
public:
  Pointer(Any* raw);
  Pointer(const Pointer<Any>& o) = default;
  Pointer(Pointer<Any>&& o) = default;

  Pointer<Any>& operator=(const Pointer<Any>& o) = default;
  Pointer<Any>& operator=(Pointer<Any>&& o) = default;
  Pointer<Any>& operator=(Any* raw);

  bool isNull() const;
  Any* get();
  Any* const get() const;

//protected:
  /**
   * Raw pointer.
   */
  Any* raw;

  /**
   * Generation.
   */
  size_t gen;
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
bi::Pointer<T>::operator U() {
  /* conversion operators in generated code are marked explicit, so the
   * cast is necessary here */
  return static_cast<U>(*get());
}

template<class T>
template<class U, typename >
bi::Pointer<T>::operator U() const {
  return static_cast<U>(*get());
}

template<class T>
T* bi::Pointer<T>::get() {
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
T* const bi::Pointer<T>::get() const {
#ifdef NDEBUG
  return static_cast<T*>(Pointer<Any>::get());
#else
  auto raw = Pointer<Any>::get();
  if (raw) {
    auto result = dynamic_cast<T* const>(raw);
    assert(result);
    return result;
  } else {
    return nullptr;
  }
#endif
}

template<class T>
template<class U>
boost::optional<bi::Pointer<U>> bi::Pointer<T>::cast() const {
  boost::optional<bi::Pointer<U>> pointer;
  U* raw1 = dynamic_cast<U*>(this->raw);
  if (raw1) {
    pointer = raw1;
  }
  return pointer;
}
