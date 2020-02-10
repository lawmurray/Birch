/**
 * @file
 */
#pragma once

#include "libbirch/class.hpp"
#include "libbirch/Counted.hpp"

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
  using super_type = InitPtr<typename bi::type::super_type<value_type>::type>;
  using shared_type = SharedPtr<T>;
  using weak_type = WeakPtr<T>;
  using init_type = InitPtr<T>;

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
class InitPtr<Counted> {
  template<class U> friend class SharedPtr;
  template<class U> friend class WeakPtr;
  template<class U> friend class InitPtr;
public:
  using value_type = Counted;
  using shared_type = SharedPtr<value_type>;
  using weak_type = WeakPtr<value_type>;
  using init_type = InitPtr<value_type>;

  /**
   * Constructor.
   */
  explicit InitPtr(Counted* ptr = nullptr);

  /**
   * Destructor.
   */
  ~InitPtr();

  /**
   * Get the raw pointer.
   */
  Counted* get() const;

  /**
   * Get the raw pointer as const.
   */
  Counted* pull() const;

  /**
   * Replace.
   */
  void replace(Counted* ptr);

  /**
   * Release.
   */
  void release();

  /**
   * Dereference.
   */
  Counted& operator*() const;

  /**
   * Member access.
   */
  Counted* operator->() const;

  /**
   * Equal comparison.
   */
  bool operator==(const SharedPtr<Counted>& o) const;
  bool operator==(const WeakPtr<Counted>& o) const;
  bool operator==(const InitPtr<Counted>& o) const;
  bool operator==(const Counted* o) const;

  /**
   * Not equal comparison.
   */
  bool operator!=(const SharedPtr<Counted>& o) const;
  bool operator!=(const WeakPtr<Counted>& o) const;
  bool operator!=(const InitPtr<Counted>& o) const;
  bool operator!=(const Counted* o) const;

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
  Counted* ptr;
};
}

#include "libbirch/SharedPtr.hpp"
#include "libbirch/WeakPtr.hpp"

libbirch::InitPtr<libbirch::Counted>::InitPtr(Counted* ptr) :
    ptr(ptr) {
  //
}

libbirch::InitPtr<libbirch::Counted>::~InitPtr() {
  release();
}

libbirch::Counted* libbirch::InitPtr<libbirch::Counted>::get() const {
  return ptr;
}

libbirch::Counted* libbirch::InitPtr<libbirch::Counted>::pull() const {
  return ptr;
}

void libbirch::InitPtr<libbirch::Counted>::replace(Counted* ptr) {
  this->ptr = ptr;
}

void libbirch::InitPtr<libbirch::Counted>::release() {
  ptr = nullptr;
}

libbirch::Counted& libbirch::InitPtr<libbirch::Counted>::operator*() const {
  return *get();
}

libbirch::Counted* libbirch::InitPtr<libbirch::Counted>::operator->() const {
  return get();
}

bool libbirch::InitPtr<libbirch::Counted>::operator==(
    const SharedPtr<Counted>& o) const {
  return ptr == o.ptr;
}

bool libbirch::InitPtr<libbirch::Counted>::operator==(
    const WeakPtr<Counted>& o) const {
  return ptr == o.ptr;
}

bool libbirch::InitPtr<libbirch::Counted>::operator==(
    const InitPtr<Counted>& o) const {
  return ptr == o.ptr;
}

bool libbirch::InitPtr<libbirch::Counted>::operator==(const Counted* o) const {
  return ptr == o;
}

bool libbirch::InitPtr<libbirch::Counted>::operator!=(
    const SharedPtr<Counted>& o) const {
  return ptr != o.ptr;
}

bool libbirch::InitPtr<libbirch::Counted>::operator!=(
    const WeakPtr<Counted>& o) const {
  return ptr != o.ptr;
}

bool libbirch::InitPtr<libbirch::Counted>::operator!=(
    const InitPtr<Counted>& o) const {
  return ptr != o.ptr;
}

bool libbirch::InitPtr<libbirch::Counted>::operator!=(const Counted* o) const {
  return ptr != o;
}

libbirch::InitPtr<libbirch::Counted>::operator bool() const {
  return ptr != nullptr;
}

template<class U>
U libbirch::InitPtr<libbirch::Counted>::dynamic_pointer_cast() const {
  U cast;
  cast.replace(dynamic_cast<typename U::value_type*>(ptr));
  return cast;
}

template<class U>
U libbirch::InitPtr<libbirch::Counted>::static_pointer_cast() const {
  U cast;
  cast.replace(static_cast<typename U::value_type*>(ptr));
  return cast;
}
