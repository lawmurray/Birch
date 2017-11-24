/**
 * @file
 */
#pragma once

#include "libbirch/global.hpp"

#include <cstdint>

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
   * Constructor.
   */
  Pointer(T* raw, const size_t gen);

  /**
   * Generic comparison operator.
   */
  template<class U>
  bool operator!=(const Pointer<U>& o) {
    return this->raw != o.raw || this->gen != o.gen;
  }

  /**
   * Generic comparison operator.
   */
  template<class U>
  bool operator==(const Pointer<U>& o) {
    return this->raw == o.raw && this->gen == o.gen;
  }

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
  T* get() const;

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
  auto operator()(Args ... args) const {
    return (*get())(args...);
  }
};

template<>
class Pointer<Any> {
  friend class std::hash<bi::Pointer<bi::Any>>;
  friend class std::equal_to<bi::Pointer<bi::Any>>;
public:
  Pointer(Any* raw = nullptr);
  Pointer(Any* raw, const size_t gen);
  Pointer<Any>& operator=(Any* raw);

  bool isNull() const;
  Any* get() const;

  /**
   * Generation.
   */
  size_t gen;

protected:
  /**
   * Raw pointer.
   */
  Any* raw;
};
}

namespace std {
template<>
struct hash<bi::Pointer<bi::Any>> : public std::hash<bi::Any*> {
  size_t operator()(const bi::Pointer<bi::Any>& o) const {
    /* the generation is ignored in the hash, as it is reasonably unlikely
     * for two pointers with the same raw pointer but different generation to
     * occur in the same allocation map; this only occurs if memory is
     * garbage collected and reused within the same fiber */
    return std::hash<bi::Any*>::operator()(o.raw);
  }
};

template<>
struct equal_to<bi::Pointer<bi::Any>> {
  bool operator()(const bi::Pointer<bi::Any>& o1,
      const bi::Pointer<bi::Any>& o2) const {
    return o1.raw == o2.raw && o1.gen == o2.gen;
  }
};
}

template<class T>
bi::Pointer<T>::Pointer(T* raw) :
    super_type(raw) {
  //
}

template<class T>
bi::Pointer<T>::Pointer(T* raw, const size_t gen) :
    super_type(raw, gen) {
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
