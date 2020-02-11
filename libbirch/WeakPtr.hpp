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
  using super_type = WeakPtr<typename bi::type::super_type<value_type>::type>;
  using shared_type = SharedPtr<value_type>;
  using weak_type = WeakPtr<value_type>;
  using init_type = InitPtr<value_type>;

  /**
   * Constructor.
   */
  WeakPtr(T* ptr = nullptr) :
      super_type(ptr) {
    //
  }

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
class WeakPtr<Counted> {
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
  WeakPtr(Counted* ptr = nullptr);

  /**
   * Shared constructor.
   */
  WeakPtr(const SharedPtr<Counted>& o);

  /**
   * Init constructor.
   */
  WeakPtr(const InitPtr<Counted>& o);

  /**
   * Copy constructor.
   */
  WeakPtr(const WeakPtr<Counted>& o);

  /**
   * Move constructor.
   */
  WeakPtr(WeakPtr<Counted> && o);

  /**
   * Destructor.
   */
  ~WeakPtr();

  /**
   * Copy assignment.
   */
  WeakPtr<Counted>& operator=(const WeakPtr<Counted>& o);

  /**
   * Move assignment.
   */
  WeakPtr<Counted>& operator=(WeakPtr<Counted> && o);

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
#include "libbirch/InitPtr.hpp"

libbirch::WeakPtr<libbirch::Counted>::WeakPtr(Counted* ptr) :
    ptr(ptr) {
  if (ptr) {
    ptr->incWeak();
  }
}

libbirch::WeakPtr<libbirch::Counted>::WeakPtr(const SharedPtr<Counted>& o) :
    ptr(o.ptr) {
  if (ptr) {
    ptr->incWeak();
  }
}

libbirch::WeakPtr<libbirch::Counted>::WeakPtr(const InitPtr<Counted>& o) :
    ptr(o.ptr) {
  if (ptr) {
    ptr->incWeak();
  }
}

libbirch::WeakPtr<libbirch::Counted>::WeakPtr(const WeakPtr<Counted>& o) :
    ptr(o.ptr) {
  if (ptr) {
    assert(ptr->numWeak() > 0);
    ptr->incWeak();
  }
}

libbirch::WeakPtr<libbirch::Counted>::WeakPtr(WeakPtr<Counted> && o) :
    ptr(o.ptr) {
  o.ptr = nullptr;
}

libbirch::WeakPtr<libbirch::Counted>::~WeakPtr() {
  release();
}

libbirch::WeakPtr<libbirch::Counted>& libbirch::WeakPtr<libbirch::Counted>::operator=(
    const WeakPtr<Counted>& o) {
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

libbirch::WeakPtr<libbirch::Counted>& libbirch::WeakPtr<libbirch::Counted>::operator=(
    WeakPtr<Counted> && o) {
  auto old = ptr;
  ptr = o.ptr;
  o.ptr = nullptr;
  if (old) {
    old->decWeak();
  }
  return *this;
}

libbirch::Counted* libbirch::WeakPtr<libbirch::Counted>::get() const {
  assert(!ptr || ptr->numWeak() > 0);
  return ptr;
}

libbirch::Counted* libbirch::WeakPtr<libbirch::Counted>::pull() const {
  assert(!ptr || ptr->numWeak() > 0);
  return ptr;
}

void libbirch::WeakPtr<libbirch::Counted>::replace(Counted* ptr) {
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

void libbirch::WeakPtr<libbirch::Counted>::release() {
  if (ptr) {
    ptr->decWeak();
    ptr = nullptr;
  }
}

libbirch::Counted& libbirch::WeakPtr<libbirch::Counted>::operator*() const {
  return *get();
}

libbirch::Counted* libbirch::WeakPtr<libbirch::Counted>::operator->() const {
  return get();
}

bool libbirch::WeakPtr<libbirch::Counted>::operator==(
    const SharedPtr<Counted>& o) const {
  return ptr == o.ptr;
}

bool libbirch::WeakPtr<libbirch::Counted>::operator==(
    const WeakPtr<Counted>& o) const {
  return ptr == o.ptr;
}

bool libbirch::WeakPtr<libbirch::Counted>::operator==(
    const InitPtr<Counted>& o) const {
  return ptr == o.ptr;
}

bool libbirch::WeakPtr<libbirch::Counted>::operator==(const Counted* o) const {
  return ptr == o;
}

bool libbirch::WeakPtr<libbirch::Counted>::operator!=(
    const SharedPtr<Counted>& o) const {
  return ptr != o.ptr;
}

bool libbirch::WeakPtr<libbirch::Counted>::operator!=(
    const WeakPtr<Counted>& o) const {
  return ptr != o.ptr;
}

bool libbirch::WeakPtr<libbirch::Counted>::operator!=(
    const InitPtr<Counted>& o) const {
  return ptr != o.ptr;
}

bool libbirch::WeakPtr<libbirch::Counted>::operator!=(const Counted* o) const {
  return ptr != o;
}

libbirch::WeakPtr<libbirch::Counted>::operator bool() const {
  return ptr != nullptr;
}

template<class U>
U libbirch::WeakPtr<libbirch::Counted>::dynamic_pointer_cast() const {
  U cast;
  cast.replace(dynamic_cast<typename U::value_type*>(ptr));
  return cast;
}

template<class U>
U libbirch::WeakPtr<libbirch::Counted>::static_pointer_cast() const {
  U cast;
  cast.replace(static_cast<typename U::value_type*>(ptr));
  return cast;
}
