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
 * Shared pointer with intrusive implementation.
 *
 * @ingroup libbirch
 *
 * @tparam T Type, must derive from Counted.
 */
template<class T>
class SharedPtr: public SharedPtr<typename bi::type::super_type<T>::type> {
public:
  using value_type = T;
  using super_type = SharedPtr<typename bi::type::super_type<value_type>::type>;
  using shared_type = SharedPtr<value_type>;
  using weak_type = WeakPtr<value_type>;
  using init_type = InitPtr<value_type>;

  /**
   * Constructor.
   */
  explicit SharedPtr(T* ptr = nullptr) :
      super_type(ptr) {
    //
  }

  /**
   * Weak constructor.
   */
  SharedPtr(const WeakPtr<T>& o) :
      super_type(o) {
    //
  }

  /**
   * Init constructor.
   */
  SharedPtr(const InitPtr<T>& o) :
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
 * Shared pointer with intrusive implementation.
 *
 * @ingroup libbirch
 */
template<>
class SharedPtr<Counted> {
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
  SharedPtr(Counted* ptr = nullptr);

  /**
   * Weak constructor.
   */
  SharedPtr(const WeakPtr<Counted>& o);

  /**
   * Init constructor.
   */
  SharedPtr(const InitPtr<Counted>& o);

  /**
   * Copy constructor.
   */
  SharedPtr(const SharedPtr<Counted>& o);

  /**
   * Move constructor.
   */
  SharedPtr(SharedPtr<Counted> && o);

  /**
   * Destructor.
   */
  ~SharedPtr();

  /**
   * Copy assignment.
   */
  SharedPtr<Counted>& operator=(const SharedPtr<Counted>& o);

  /**
   * Move assignment.
   */
  SharedPtr<Counted>& operator=(SharedPtr<Counted> && o);

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

#include "libbirch/WeakPtr.hpp"
#include "libbirch/InitPtr.hpp"

libbirch::SharedPtr<libbirch::Counted>::SharedPtr(Counted* ptr) :
    ptr(ptr) {
  if (ptr) {
    ptr->init();
  }
}

libbirch::SharedPtr<libbirch::Counted>::SharedPtr(const WeakPtr<Counted>& o) :
    ptr(o.ptr) {
  if (ptr) {
    assert(ptr->numShared() > 0);
    ptr->incShared();
  }
}

libbirch::SharedPtr<libbirch::Counted>::SharedPtr(const InitPtr<Counted>& o) :
    ptr(o.ptr) {
  if (ptr) {
    assert(ptr->numShared() > 0);
    ptr->incShared();
  }
}

libbirch::SharedPtr<libbirch::Counted>::SharedPtr(const SharedPtr<Counted>& o) :
    ptr(o.ptr) {
  if (ptr) {
    ptr->incShared();
  }
}

libbirch::SharedPtr<libbirch::Counted>::SharedPtr(SharedPtr<Counted> && o) :
    ptr(o.ptr) {
  o.ptr = nullptr;
}

libbirch::SharedPtr<libbirch::Counted>::~SharedPtr() {
  release();
}

libbirch::SharedPtr<libbirch::Counted>& libbirch::SharedPtr<libbirch::Counted>::operator=(
    const SharedPtr<Counted>& o) {
  if (o.ptr) {
    o.ptr->incShared();
  }
  auto old = ptr;
  ptr = o.ptr;
  if (old) {
    old->decShared();
  }
  return *this;
}

libbirch::SharedPtr<libbirch::Counted>& libbirch::SharedPtr<libbirch::Counted>::operator=(
    SharedPtr<Counted> && o) {
  auto old = ptr;
  ptr = o.ptr;
  o.ptr = nullptr;
  if (old) {
    old->decShared();
  }
  return *this;
}

libbirch::Counted* libbirch::SharedPtr<libbirch::Counted>::get() const {
  assert(!ptr || ptr->numShared() > 0);
  return ptr;
}

libbirch::Counted* libbirch::SharedPtr<libbirch::Counted>::pull() const {
  assert(!ptr || ptr->numShared() > 0);
  return ptr;
}

void libbirch::SharedPtr<libbirch::Counted>::replace(Counted* ptr) {
  //assert(!ptr || ptr->numShared() > 0);
  auto old = this->ptr;
  if (ptr) {
    ptr->incShared();
  }
  this->ptr = ptr;
  if (old) {
    old->decShared();
  }
}

void libbirch::SharedPtr<libbirch::Counted>::release() {
  if (ptr) {
    ptr->decShared();
    ptr = nullptr;
  }
}

libbirch::Counted& libbirch::SharedPtr<libbirch::Counted>::operator*() const {
  return *get();
}

libbirch::Counted* libbirch::SharedPtr<libbirch::Counted>::operator->() const {
  return get();
}

bool libbirch::SharedPtr<libbirch::Counted>::operator==(
    const SharedPtr<Counted>& o) const {
  return ptr == o.ptr;
}

bool libbirch::SharedPtr<libbirch::Counted>::operator==(
    const WeakPtr<Counted>& o) const {
  return ptr == o.ptr;
}

bool libbirch::SharedPtr<libbirch::Counted>::operator==(
    const InitPtr<Counted>& o) const {
  return ptr == o.ptr;
}

bool libbirch::SharedPtr<libbirch::Counted>::operator==(const Counted* o) const {
  return ptr == o;
}

bool libbirch::SharedPtr<libbirch::Counted>::operator!=(
    const SharedPtr<Counted>& o) const {
  return ptr != o.ptr;
}

bool libbirch::SharedPtr<libbirch::Counted>::operator!=(
    const WeakPtr<Counted>& o) const {
  return ptr != o.ptr;
}

bool libbirch::SharedPtr<libbirch::Counted>::operator!=(
    const InitPtr<Counted>& o) const {
  return ptr != o.ptr;
}

bool libbirch::SharedPtr<libbirch::Counted>::operator!=(const Counted* o) const {
  return ptr != o;
}

libbirch::SharedPtr<libbirch::Counted>::operator bool() const {
  return ptr != nullptr;
}

template<class U>
U libbirch::SharedPtr<libbirch::Counted>::dynamic_pointer_cast() const {
  U cast;
  cast.replace(dynamic_cast<typename U::value_type*>(ptr));
  return cast;
}

template<class U>
U libbirch::SharedPtr<libbirch::Counted>::static_pointer_cast() const {
  U cast;
  cast.replace(static_cast<typename U::value_type*>(ptr));
  return cast;
}
