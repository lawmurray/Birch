/**
 * @file
 */
#pragma once

#include "libbirch/class.hpp"
#include "libbirch/Any.hpp"

namespace libbirch {
template<class T> class SharedPtr;
template<class T> class WeakPtr;
template<class T> class InitPtr;

/**
 * Smart pointer that does not update reference counts, but that does
 * initialize to nullptr.
 *
 * @ingroup libbirch
 *
 * @tparam T Type, must derive from Counted.
 */
template<class T>
class InitPtr: public InitPtr<typename bi::type::super_type<T>::type> {
public:
  using value_type = T;
  using super_type = InitPtr<typename bi::type::super_type<T>::type>;

  /**
   * Constructor.
   */
  explicit InitPtr(T* ptr = nullptr) :
      super_type(ptr) {
    //
  }

  /**
   * Get the raw pointer.
   */
  T* get() const {
    return static_cast<T*>(super_type::get());
  }

  /**
   * Get the raw pointer as const.
   */
  T* pull() const {
    return static_cast<T*>(super_type::pull());
  }

  /**
   * Dereference.
   */
  T& operator*() const {
    return *get();
  }

  /**
   * Member access.
   */
  T* operator->() const {
    return get();
  }
};

/**
 * Smart pointer that does not update reference counts, but that does
 * initialize to nullptr.
 *
 * @ingroup libbirch
 */
template<>
class InitPtr<Any> {
  template<class U> friend class SharedPtr;
  template<class U> friend class WeakPtr;
  template<class U> friend class InitPtr;
public:
  using value_type = Any;

  /**
   * Constructor.
   */
  explicit InitPtr(Any* ptr = nullptr);

  /**
   * Destructor.
   */
  ~InitPtr();

  /**
   * Get the raw pointer.
   */
  Any* get() const;

  /**
   * Get the raw pointer as const.
   */
  Any* pull() const;

  /**
   * Replace.
   */
  void replace(Any* ptr);

  /**
   * Release.
   */
  void release();

  /**
   * Dereference.
   */
  Any& operator*() const;

  /**
   * Member access.
   */
  Any* operator->() const;

  /**
   * Equal comparison.
   */
  bool operator==(const SharedPtr<Any>& o) const;
  bool operator==(const WeakPtr<Any>& o) const;
  bool operator==(const InitPtr<Any>& o) const;
  bool operator==(const Any* o) const;

  /**
   * Not equal comparison.
   */
  bool operator!=(const SharedPtr<Any>& o) const;
  bool operator!=(const WeakPtr<Any>& o) const;
  bool operator!=(const InitPtr<Any>& o) const;
  bool operator!=(const Any* o) const;

  /**
   * Is the pointer not null?
   */
  operator bool() const;

  /**
   * Dynamic cast.
   */
  template<class U>
  U dynamic_pointer_cast() const;

  /**
   * Static cast.
   */
  template<class U>
  U static_pointer_cast() const;

private:
  /**
   * Raw pointer.
   */
  Any* ptr;
};
}

#include "libbirch/SharedPtr.hpp"
#include "libbirch/WeakPtr.hpp"

libbirch::InitPtr<libbirch::Any>::InitPtr(Any* ptr) :
    ptr(ptr) {
  //
}

libbirch::InitPtr<libbirch::Any>::~InitPtr() {
  release();
}

libbirch::Any* libbirch::InitPtr<libbirch::Any>::get() const {
  return ptr;
}

libbirch::Any* libbirch::InitPtr<libbirch::Any>::pull() const {
  return ptr;
}

void libbirch::InitPtr<libbirch::Any>::replace(Any* ptr) {
  this->ptr = ptr;
}

void libbirch::InitPtr<libbirch::Any>::release() {
  ptr = nullptr;
}

libbirch::Any& libbirch::InitPtr<libbirch::Any>::operator*() const {
  return *get();
}

libbirch::Any* libbirch::InitPtr<libbirch::Any>::operator->() const {
  return get();
}

bool libbirch::InitPtr<libbirch::Any>::operator==(
    const SharedPtr<Any>& o) const {
  return ptr == o.ptr;
}

bool libbirch::InitPtr<libbirch::Any>::operator==(
    const WeakPtr<Any>& o) const {
  return ptr == o.ptr;
}

bool libbirch::InitPtr<libbirch::Any>::operator==(
    const InitPtr<Any>& o) const {
  return ptr == o.ptr;
}

bool libbirch::InitPtr<libbirch::Any>::operator==(const Any* o) const {
  return ptr == o;
}

bool libbirch::InitPtr<libbirch::Any>::operator!=(
    const SharedPtr<Any>& o) const {
  return ptr != o.ptr;
}

bool libbirch::InitPtr<libbirch::Any>::operator!=(
    const WeakPtr<Any>& o) const {
  return ptr != o.ptr;
}

bool libbirch::InitPtr<libbirch::Any>::operator!=(
    const InitPtr<Any>& o) const {
  return ptr != o.ptr;
}

bool libbirch::InitPtr<libbirch::Any>::operator!=(const Any* o) const {
  return ptr != o;
}

libbirch::InitPtr<libbirch::Any>::operator bool() const {
  return ptr != nullptr;
}

template<class U>
U libbirch::InitPtr<libbirch::Any>::dynamic_pointer_cast() const {
  U cast;
  cast.replace(dynamic_cast<typename U::value_type*>(ptr));
  return cast;
}

template<class U>
U libbirch::InitPtr<libbirch::Any>::static_pointer_cast() const {
  U cast;
  cast.replace(static_cast<typename U::value_type*>(ptr));
  return cast;
}
