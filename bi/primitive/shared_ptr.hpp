/**
 * @file
 */
#pragma once

#include <memory>

namespace bi {
/**
 * Smart pointer reflecting shared ownership of the object associated with a raw
 * pointer. Unlike other smart pointers, allows assignment to a raw
 * pointer.
 *
 * @tparam T Type.
 */
template<class T>
class shared_ptr {
  template<class U>
  friend class shared_ptr;

  template<class U>
  friend const shared_ptr<U>& cast(const shared_ptr<T>&);
public:
  /**
   * Constructor. The pointer is initialised to null.
   */
  shared_ptr();

  /**
   * Construct from raw pointer.
   */
  shared_ptr(T* ptr);

  /**
   * Construct from STL smart pointer.
   */
  shared_ptr(const std::shared_ptr<T>& ptr);

  /**
   * Copy constructor.
   */
  shared_ptr(const shared_ptr<T>& o);

  /**
   * Destructor.
   */
  ~shared_ptr();

  /**
   * Assignment of raw pointer.
   */
  shared_ptr<T>& operator=(T* o);

  /**
   * Cast to shared pointer of base or derived type.
   */
  template<class U>
  operator shared_ptr<U>() const;

  /*
   * Equality operators.
   */
  bool operator==(const shared_ptr<T>& o) const {
    return get() == o.get();
  }
  bool operator!=(const shared_ptr<T>& o) const {
    return get() != o.get();
  }

  /*
   * Inequality operators.
   */
  bool operator<(const shared_ptr<T>& o) const {
    return get() < o.get();
  }
  bool operator>(const shared_ptr<T>& o) const {
    return get() > o.get();
  }
  bool operator<=(const shared_ptr<T>& o) const {
    return get() <= o.get();
  }
  bool operator>=(const shared_ptr<T>& o) const {
    return get() >= o.get();
  }

  /**
   * Cast to bool (check for null pointer).
   */
  operator bool() const;

  /**
   * Dereference.
   */
  T& operator*() const;

  /**
   * Member access.
   */
  T* operator->() const;

  /**
   * Get raw pointer.
   */
  T* get() const;

private:
  /**
   * Raw pointer.
   */
  std::shared_ptr<T> ptr;
};

template<class T, class ...Args>
shared_ptr<T> make_shared(Args ... args) {
  return shared_ptr<T>(std::make_shared<T>(args...));
}
}

template<class T>
bi::shared_ptr<T>::shared_ptr() :
    ptr(nullptr) {
  //
}

template<class T>
bi::shared_ptr<T>::shared_ptr(T* ptr) :
    ptr(ptr) {
  //
}

template<class T>
bi::shared_ptr<T>::shared_ptr(const std::shared_ptr<T>& ptr) :
    ptr(ptr) {
  //
}

template<class T>
bi::shared_ptr<T>::shared_ptr(const shared_ptr<T>& o) :
    ptr(o.ptr) {
  //
}

template<class T>
bi::shared_ptr<T>::~shared_ptr() {
  //
}

template<class T>
inline bi::shared_ptr<T>& bi::shared_ptr<T>::operator=(T* ptr) {
  if (this->ptr.get() != ptr) {
    this->ptr.reset(ptr);
  }
  return *this;
}

template<class T>
template<class U>
bi::shared_ptr<T>::operator shared_ptr<U>() const {
  return std::dynamic_pointer_cast<U>(ptr);
}

template<class T>
inline bi::shared_ptr<T>::operator bool() const {
  return ptr.get() != nullptr;
}

template<class T>
inline T& bi::shared_ptr<T>::operator*() const {
  return *ptr;
}

template<class T>
inline T* bi::shared_ptr<T>::operator->() const {
  return ptr.get();
}

template<class T>
inline T* bi::shared_ptr<T>::get() const {
  return ptr.get();
}
