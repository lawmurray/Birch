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
  using super_type = SharedPtr<typename bi::type::super_type<T>::type>;

  /**
   * Constructor. This is intended for use immediately after construction of
   * the object; the reference count is not incremented, as it should be
   * initialized accordingly.
   */
  explicit SharedPtr(T* ptr = nullptr) :
      super_type(ptr) {
    //
  }

  /**
   * Constructor with in-place construction of the referent.
   */
  template<class ... Args>
  SharedPtr(Label* context, Args ... args) :
      SharedPtr(new T(context, args...)) {
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
class SharedPtr<Any> {
  template<class U> friend class SharedPtr;
  template<class U> friend class WeakPtr;
  template<class U> friend class InitPtr;
public:
  using value_type = Any;

  /**
   * Constructor. This is intended for use immediately after construction of
   * the object; the reference count is not incremented, as it should be
   * initialized accordingly.
   */
  explicit SharedPtr(Any* ptr = nullptr);

  /**
   * Weak constructor.
   */
  SharedPtr(const WeakPtr<Any>& o);

  /**
   * Init constructor.
   */
  SharedPtr(const InitPtr<Any>& o);

  /**
   * Copy constructor.
   */
  SharedPtr(const SharedPtr<Any>& o);

  /**
   * Move constructor.
   */
  SharedPtr(SharedPtr<Any> && o);

  /**
   * Destructor.
   */
  ~SharedPtr();

  /**
   * Copy assignment.
   */
  SharedPtr<Any>& operator=(const SharedPtr<Any>& o);

  /**
   * Move assignment.
   */
  SharedPtr<Any>& operator=(SharedPtr<Any> && o);

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

#include "libbirch/WeakPtr.hpp"
#include "libbirch/InitPtr.hpp"

libbirch::SharedPtr<libbirch::Any>::SharedPtr(Any* ptr) :
    ptr(ptr) {
  if (ptr) {
    ptr->init();
  }
}

libbirch::SharedPtr<libbirch::Any>::SharedPtr(const WeakPtr<Any>& o) :
    ptr(o.ptr) {
  if (ptr) {
    assert(ptr->numShared() > 0);
    ptr->incShared();
  }
}

libbirch::SharedPtr<libbirch::Any>::SharedPtr(const InitPtr<Any>& o) :
    ptr(o.ptr) {
  if (ptr) {
    assert(ptr->numShared() > 0);
    ptr->incShared();
  }
}

libbirch::SharedPtr<libbirch::Any>::SharedPtr(const SharedPtr<Any>& o) :
    ptr(o.ptr) {
  if (ptr) {
    ptr->incShared();
  }
}

libbirch::SharedPtr<libbirch::Any>::SharedPtr(SharedPtr<Any> && o) :
    ptr(o.ptr) {
  o.ptr = nullptr;
}

libbirch::SharedPtr<libbirch::Any>::~SharedPtr() {
  release();
}

libbirch::SharedPtr<libbirch::Any>& libbirch::SharedPtr<libbirch::Any>::operator=(
    const SharedPtr<Any>& o) {
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

libbirch::SharedPtr<libbirch::Any>& libbirch::SharedPtr<libbirch::Any>::operator=(
    SharedPtr<Any> && o) {
  auto old = ptr;
  ptr = o.ptr;
  o.ptr = nullptr;
  if (old) {
    old->decShared();
  }
  return *this;
}

libbirch::Any* libbirch::SharedPtr<libbirch::Any>::get() const {
  assert(!ptr || ptr->numShared() > 0);
  return ptr;
}

libbirch::Any* libbirch::SharedPtr<libbirch::Any>::pull() const {
  assert(!ptr || ptr->numShared() > 0);
  return ptr;
}

void libbirch::SharedPtr<libbirch::Any>::replace(Any* ptr) {
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

void libbirch::SharedPtr<libbirch::Any>::release() {
  if (ptr) {
    ptr->decShared();
    ptr = nullptr;
  }
}

libbirch::Any& libbirch::SharedPtr<libbirch::Any>::operator*() const {
  return *get();
}

libbirch::Any* libbirch::SharedPtr<libbirch::Any>::operator->() const {
  return get();
}

bool libbirch::SharedPtr<libbirch::Any>::operator==(
    const SharedPtr<Any>& o) const {
  return ptr == o.ptr;
}

bool libbirch::SharedPtr<libbirch::Any>::operator==(
    const WeakPtr<Any>& o) const {
  return ptr == o.ptr;
}

bool libbirch::SharedPtr<libbirch::Any>::operator==(
    const InitPtr<Any>& o) const {
  return ptr == o.ptr;
}

bool libbirch::SharedPtr<libbirch::Any>::operator==(const Any* o) const {
  return ptr == o;
}

bool libbirch::SharedPtr<libbirch::Any>::operator!=(
    const SharedPtr<Any>& o) const {
  return ptr != o.ptr;
}

bool libbirch::SharedPtr<libbirch::Any>::operator!=(
    const WeakPtr<Any>& o) const {
  return ptr != o.ptr;
}

bool libbirch::SharedPtr<libbirch::Any>::operator!=(
    const InitPtr<Any>& o) const {
  return ptr != o.ptr;
}

bool libbirch::SharedPtr<libbirch::Any>::operator!=(const Any* o) const {
  return ptr != o;
}

libbirch::SharedPtr<libbirch::Any>::operator bool() const {
  return ptr != nullptr;
}

template<class U>
U libbirch::SharedPtr<libbirch::Any>::dynamic_pointer_cast() const {
  U cast;
  cast.replace(dynamic_cast<typename U::value_type*>(ptr));
  return cast;
}

template<class U>
U libbirch::SharedPtr<libbirch::Any>::static_pointer_cast() const {
  U cast;
  cast.replace(static_cast<typename U::value_type*>(ptr));
  return cast;
}
