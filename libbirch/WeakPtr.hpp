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
 * Weak pointer with intrusive implementation.
 *
 * @ingroup libbirch
 *
 * @tparam T Type, must derive from Counted.
 */
template<class T>
class WeakPtr: public WeakPtr<typename bi::type::super_type<T>::type> {
public:
  using value_type = T;
  using super_type = WeakPtr<typename bi::type::super_type<T>::type>;

  /**
   * Shared constructor.
   */
  WeakPtr(const SharedPtr<T>& o) :
      super_type(o) {
    //
  }

  /**
   * Init constructor.
   */
  WeakPtr(const InitPtr<T>& o) :
      super_type(o) {
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
 * Weak pointer with intrusive implementation.
 *
 * @ingroup libbirch
 */
template<>
class WeakPtr<Any> {
  template<class U> friend class SharedPtr;
  template<class U> friend class WeakPtr;
  template<class U> friend class InitPtr;
public:
  using value_type = Any;

  /**
   * Constructor.
   */
  WeakPtr();

  /**
   * Shared constructor.
   */
  WeakPtr(const SharedPtr<Any>& o);

  /**
   * Init constructor.
   */
  WeakPtr(const InitPtr<Any>& o);

  /**
   * Copy constructor.
   */
  WeakPtr(const WeakPtr<Any>& o);

  /**
   * Move constructor.
   */
  WeakPtr(WeakPtr<Any> && o);

  /**
   * Destructor.
   */
  ~WeakPtr();

  /**
   * Copy assignment.
   */
  WeakPtr<Any>& operator=(const WeakPtr<Any>& o);

  /**
   * Move assignment.
   */
  WeakPtr<Any>& operator=(WeakPtr<Any> && o);

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
#include "libbirch/InitPtr.hpp"

libbirch::WeakPtr<libbirch::Any>::WeakPtr() :
    ptr(nullptr) {
  //
}

libbirch::WeakPtr<libbirch::Any>::WeakPtr(const SharedPtr<Any>& o) :
    ptr(o.ptr) {
  if (ptr) {
    ptr->incWeak();
  }
}

libbirch::WeakPtr<libbirch::Any>::WeakPtr(const InitPtr<Any>& o) :
    ptr(o.ptr) {
  if (ptr) {
    ptr->incWeak();
  }
}

libbirch::WeakPtr<libbirch::Any>::WeakPtr(const WeakPtr<Any>& o) :
    ptr(o.ptr) {
  if (ptr) {
    assert(ptr->numWeak() > 0);
    ptr->incWeak();
  }
}

libbirch::WeakPtr<libbirch::Any>::WeakPtr(WeakPtr<Any> && o) :
    ptr(o.ptr) {
  o.ptr = nullptr;
}

libbirch::WeakPtr<libbirch::Any>::~WeakPtr() {
  release();
}

libbirch::WeakPtr<libbirch::Any>& libbirch::WeakPtr<libbirch::Any>::operator=(
    const WeakPtr<Any>& o) {
  if (o.ptr) {
    o.ptr->incWeak();
  }
  auto old = ptr;
  ptr = o.ptr;
  if (old) {
    old->decWeak();
  }
  return *this;
}

libbirch::WeakPtr<libbirch::Any>& libbirch::WeakPtr<libbirch::Any>::operator=(
    WeakPtr<Any> && o) {
  auto old = ptr;
  ptr = o.ptr;
  o.ptr = nullptr;
  if (old) {
    old->decWeak();
  }
  return *this;
}

libbirch::Any* libbirch::WeakPtr<libbirch::Any>::get() const {
  assert(!ptr || ptr->numWeak() > 0);
  return ptr;
}

libbirch::Any* libbirch::WeakPtr<libbirch::Any>::pull() const {
  assert(!ptr || ptr->numWeak() > 0);
  return ptr;
}

void libbirch::WeakPtr<libbirch::Any>::replace(Any* ptr) {
  assert(!ptr || ptr->numWeak() > 0);
  auto old = this->ptr;
  if (ptr) {
    ptr->incWeak();
  }
  this->ptr = ptr;
  if (old) {
    old->decWeak();
  }
}

void libbirch::WeakPtr<libbirch::Any>::release() {
  if (ptr) {
    ptr->decWeak();
    ptr = nullptr;
  }
}

libbirch::Any& libbirch::WeakPtr<libbirch::Any>::operator*() const {
  return *get();
}

libbirch::Any* libbirch::WeakPtr<libbirch::Any>::operator->() const {
  return get();
}

bool libbirch::WeakPtr<libbirch::Any>::operator==(
    const SharedPtr<Any>& o) const {
  return ptr == o.ptr;
}

bool libbirch::WeakPtr<libbirch::Any>::operator==(
    const WeakPtr<Any>& o) const {
  return ptr == o.ptr;
}

bool libbirch::WeakPtr<libbirch::Any>::operator==(
    const InitPtr<Any>& o) const {
  return ptr == o.ptr;
}

bool libbirch::WeakPtr<libbirch::Any>::operator==(const Any* o) const {
  return ptr == o;
}

bool libbirch::WeakPtr<libbirch::Any>::operator!=(
    const SharedPtr<Any>& o) const {
  return ptr != o.ptr;
}

bool libbirch::WeakPtr<libbirch::Any>::operator!=(
    const WeakPtr<Any>& o) const {
  return ptr != o.ptr;
}

bool libbirch::WeakPtr<libbirch::Any>::operator!=(
    const InitPtr<Any>& o) const {
  return ptr != o.ptr;
}

bool libbirch::WeakPtr<libbirch::Any>::operator!=(const Any* o) const {
  return ptr != o;
}

libbirch::WeakPtr<libbirch::Any>::operator bool() const {
  return ptr != nullptr;
}

template<class U>
U libbirch::WeakPtr<libbirch::Any>::dynamic_pointer_cast() const {
  U cast;
  cast.replace(dynamic_cast<typename U::value_type*>(ptr));
  return cast;
}

template<class U>
U libbirch::WeakPtr<libbirch::Any>::static_pointer_cast() const {
  U cast;
  cast.replace(static_cast<typename U::value_type*>(ptr));
  return cast;
}
