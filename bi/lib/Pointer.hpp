/**
 * @file
 */
#pragma once

#include "bi/lib/global.hpp"
#include "bi/lib/Object.hpp"

namespace bi {
/**
 * Smart pointer fiber-local heaps, with copy-on-write semantics.
 *
 * @ingroup library
 *
 * @tparam T Type.
 */
template<class T>
class Pointer: public Pointer<typename super_type<T>::type> {
  friend class Object;
public:
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
   * Raw pointer assignment operator.
   */
  Pointer<T>& operator=(T* raw);

  /**
   * Null pointer assignment operator.
   */
  Pointer<T>& operator=(const std::nullptr_t&);

  /**
   * Generic value assignment operator.
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

protected:
  /**
   * Constructor for pointer_from_this() in Object.
   */
  Pointer(T* raw, intptr_t index);
};

template<>
class Pointer<Object> {
public:
  /**
   * Raw pointer constructor.
   */
  Pointer(Object* raw = nullptr);

  /**
   * Is this a null pointer?
   */
  bool isNull() const {
    return index < 0;
  }

protected:
  /**
   * Constructor for pointer_from_this().
   */
  Pointer(Object* raw, intptr_t index);

  /**
   * The index of the heap allocation, -1 for null.
   */
  intptr_t index;
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
  assert(fiberHeap);
  if (raw) {
    this->index = fiberHeap->put(raw);
  } else {
    this->index = -1;
  }
  return *this;
}

template<class T>
bi::Pointer<T>& bi::Pointer<T>::operator=(const std::nullptr_t&) {
  this->index = -1;
  return *this;
}

template<class T>
template<class U>
bi::Pointer<T>& bi::Pointer<T>::operator=(const U& o) {
  *get() = o;
  return *this;
}

#include <iostream>
template<class T>
T* bi::Pointer<T>::get() {
  if (this->index < 0) {
    return nullptr;
  } else {
    Object* o;
    assert(fiberHeap);
    o = fiberHeap->get(this->index);
    if (o->getGen() < fiberGen) {
      /* (possibly) shared and writeable, copy now (copy-on-write) */
      o = o->clone();
      fiberHeap->set(this->index, o);
    }
    T* raw;
    #ifdef NDEBUG
    raw = static_cast<T*>(o);
    #else
    raw = dynamic_cast<T*>(o);
    assert(raw);
    #endif
    return raw;
  }
}

template<class T>
T* const bi::Pointer<T>::get() const {
  if (this->index < 0) {
    return nullptr;
  } else {
    Object* o;
    assert(fiberHeap);
    o = fiberHeap->get(this->index);
    T* raw;
    #ifdef NDEBUG
    raw = static_cast<T*>(o);
    #else
    raw = dynamic_cast<T*>(o);
    assert(raw);
    #endif
    return raw;
  }
}

template<class T>
bi::Pointer<T>::Pointer(T* raw, intptr_t index) :
    super_type(raw, index) {
  //
}

inline bi::Pointer<bi::Object>::Pointer(Object* raw) {
  assert(fiberHeap);
  if (raw) {
    index = fiberHeap->put(raw);
  } else {
    index = -1;
  }
}

inline bi::Pointer<bi::Object>::Pointer(Object* raw, intptr_t index) {
  assert(index >= 0 || !raw);
  assert(index < 0 || fiberHeap->get(index) == raw);

  this->index = index;
}
