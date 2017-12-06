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
public:
  typedef T value_type;
  typedef SharedPointer<T> this_type;
  typedef SharedPointer<typename super_type<T>::type> super_type;

  /**
   * Constructor from raw pointer.
   */
  SharedPointer(T* raw = nullptr);

  /**
   * Constructor from allocation.
   */
  SharedPointer(Allocation* allocation);

  /**
   * Copy constructor.
   */
  SharedPointer(const SharedPointer<T>& o);

  /**
   * Copy constructor.
   */
  SharedPointer(const WeakPointer<T>& o);

  /**
   * Constructor from shared pointer.
   */
  SharedPointer(const SharedPointer<T>& o, const world_t world);

  /**
   * Constructor from weak pointer.
   */
  SharedPointer(const WeakPointer<T>& o, const world_t world);

  /**
   * Assignment from null pointer.
   */
  SharedPointer<T>& operator=(const std::nullptr_t& o);

  /**
   * Assignment from shared pointer.
   */
  SharedPointer<T>& operator=(const SharedPointer<T>& o);

  /**
   * Assignment from weak pointer.
   */
  SharedPointer<T>& operator=(const WeakPointer<T>& o);

  /**
   * Generic assignment from shared pointer.
   */
  template<class U, typename = std::enable_if<std::is_base_of<T,U>::value>>
  SharedPointer<T>& operator=(const SharedPointer<U>& o);

  /**
   * Generic assignment from weak pointer.
   */
  template<class U, typename = std::enable_if<std::is_base_of<T,U>::value>>
  SharedPointer<T>& operator=(const WeakPointer<U>& o);

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

  /**
   * Comparison operators.
   */
  template<class U>
  bool operator!=(const SharedPointer<U>& o) const {
    return this->raw != o.raw || this->gen != o.gen;
  }
  template<class U>
  bool operator==(const SharedPointer<U>& o) const {
    return this->raw == o.raw && this->gen == o.gen;
  }
};

template<>
class SharedPointer<Any> {
  friend class WeakPointer<Any> ;
public:
  SharedPointer(Any* raw = nullptr);
  SharedPointer(Allocation* allocation);
  SharedPointer(const SharedPointer<Any>& o);
  SharedPointer(const WeakPointer<Any>& o);
  SharedPointer(const SharedPointer<Any>& o, const world_t world);
  SharedPointer(const WeakPointer<Any>& o, const world_t world);
  SharedPointer<Any>& operator=(const std::nullptr_t& o);
  SharedPointer<Any>& operator=(const SharedPointer<Any>& o);
  SharedPointer<Any>& operator=(const WeakPointer<Any>& o);
  ~SharedPointer();

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

  /**
   * Reset the allocation.
   */
  void reset(Allocation* allocation);

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
   * Allocation control structure.
   */
  Allocation* allocation;
};
}

#include <cassert>

template<class T>
bi::SharedPointer<T>::SharedPointer(T* raw) :
    super_type(raw) {
  //
}

template<class T>
bi::SharedPointer<T>::SharedPointer(Allocation* allocation) :
    super_type(allocation) {
  //
}

template<class T>
bi::SharedPointer<T>::SharedPointer(const SharedPointer<T>& o) :
    super_type(o) {
  //
}

template<class T>
bi::SharedPointer<T>::SharedPointer(const WeakPointer<T>& o) :
    super_type(o) {
  //
}

template<class T>
bi::SharedPointer<T>::SharedPointer(const SharedPointer<T>& o,
    const world_t world) :
    super_type(o, world) {
  //
}

template<class T>
bi::SharedPointer<T>::SharedPointer(const WeakPointer<T>& o,
    const world_t world) :
    super_type(o, world) {
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
template<class U, typename >
bi::SharedPointer<T>& bi::SharedPointer<T>::operator=(
    const WeakPointer<U>& o) {
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
T* bi::SharedPointer<T>::get() const {
#ifdef NDEBUG
  return static_cast<T*>(SharedPointer<Any>::get());
#else
  auto raw = SharedPointer<Any>::get();
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
bi::SharedPointer<U> bi::SharedPointer<T>::dynamic_pointer_cast() const {
  if (dynamic_cast<U*>(get())) {
    return SharedPointer<U>(this->allocation);
  } else {
    return SharedPointer<U>();
  }
}

template<class T>
template<class U>
bi::SharedPointer<U> bi::SharedPointer<T>::static_pointer_cast() const {
#ifndef NDEBUG
  assert(dynamic_cast<U*>(get()));
#endif
  return SharedPointer<U>(this->allocation);
}

template<class U>
bi::SharedPointer<U> bi::SharedPointer<bi::Any>::dynamic_pointer_cast() const {
  if (dynamic_cast<U*>(get())) {
    return SharedPointer<U>(this->allocation);
  } else {
    return SharedPointer<U>();
  }
}

template<class U>
bi::SharedPointer<U> bi::SharedPointer<bi::Any>::static_pointer_cast() const {
#ifndef NDEBUG
  assert(dynamic_cast<U*>(get()));
#endif
  return SharedPointer<U>(this->allocation);
}
