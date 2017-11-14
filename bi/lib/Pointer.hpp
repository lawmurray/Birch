/**
 * @file
 */
#pragma once

#include "boost/optional.hpp"
#include "bi/lib/global.hpp"
#include "bi/lib/AllocationMap.hpp"

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
   * Assignment operator.
   */
  Pointer<T>& operator=(const Pointer<T>& o) = default;

  /**
   * Generic assignment operator.
   */
  template<class U, typename = std::enable_if<std::is_base_of<T,U>::value>>
  Pointer<T>& operator=(const Pointer<U>& o) {
    this->raw = o.raw;
    return *this;
  }

  /**
   * Raw pointer assignment operator.
   */
  Pointer<T>& operator=(T* raw);

  /**
   * Null pointer assignment operator.
   */
  Pointer<T>& operator=(const std::nullptr_t&);

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
  operator U() {
    /* conversion operators in generated code are marked explicit, so the
     * cast is necessary here */
    return static_cast<U>(*get());
  }
  template<class U, typename = std::enable_if_t<has_conversion<T,U>::value>>
  operator U() const {
    return static_cast<U>(*get());
  }

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
  /**
   * Raw pointer constructor.
   */
  Pointer(Any* raw = nullptr);

  /**
   * Is this a null pointer?
   */
  bool isNull() const {
    return raw == nullptr;
  }

//protected:
  /**
   * Raw pointer.
   */
  Any* raw;
};
}

#include "bi/lib/global.hpp"
#include "bi/lib/Fiber.hpp"

template<class T>
bi::Pointer<T>::Pointer(T* raw) :
    super_type(raw) {
  //
}

template<class T>
bi::Pointer<T>& bi::Pointer<T>::operator=(T* raw) {
  this->raw = raw;
  return *this;
}

template<class T>
bi::Pointer<T>& bi::Pointer<T>::operator=(const std::nullptr_t&) {
  this->raw = nullptr;
  return *this;
}

template<class T>
template<class U>
bi::Pointer<T>& bi::Pointer<T>::operator=(const U& o) {
  *get() = o;
  return *this;
}

template<class T>
T* bi::Pointer<T>::get() {
  if (this->isNull()) {
    return nullptr;
  } else if (this->raw->isShared()) {
    /* object is shared; it may have been cloned already via another pointer,
     * so update this pointer via the current fiber's allocation map */
    assert(fiberAllocationMap);
    this->raw = fiberAllocationMap->get(this->raw);

    /* object is writeable; if it is still shared, then clone it and add a
     * new entry to the current fiber's allocation map */
    if (this->raw->isShared()) {
      Any* to = this->raw->clone();
      fiberAllocationMap->set(this->raw, to);
      this->raw = to;
    }
  }

  /* return pointer, cast to the right type */
  T* result;
#ifdef NDEBUG
  result = static_cast<T*>(this->raw);
#else
  result = dynamic_cast<T*>(this->raw);
  assert(result);
#endif
  return result;
}

template<class T>
T* const bi::Pointer<T>::get() const {
  if (this->isNull()) {
    return nullptr;
  } else if (this->raw->isShared()) {
    /* object is shared; it may have been cloned already via another pointer,
     * so update this pointer via the current fiber's allocation map */
    assert(fiberAllocationMap);
    const_cast<Pointer<T>*>(this)->raw = fiberAllocationMap->get(this->raw);
  }

  /* return pointer, cast to the right type */
  T* result;
  #ifdef NDEBUG
  result = static_cast<T*>(this->raw);
  #else
  result = dynamic_cast<T*>(this->raw);
  assert(result);
  #endif
  return result;
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

inline bi::Pointer<bi::Any>::Pointer(Any* raw) : raw(raw) {
  //
}
