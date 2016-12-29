/**
 * @file
 */
#pragma once

#include <memory>

namespace bi {
/**
 * Smart pointer reflecting unique ownership of the object associated with a raw
 * pointer. Unlike other smart pointers, allows assignment to a raw
 * pointer.
 *
 * @tparam T Type.
 */
template<class T>
class unique_ptr {
public:
  /**
   * Constructor. The pointer is initialised to null.
   */
  unique_ptr();

  /**
   * Constructor.
   */
  unique_ptr(T* ptr);

  /**
   * Copy constructor.
   */
  unique_ptr(const unique_ptr<T>& ptr);

  /**
   * Destructor.
   */
  ~unique_ptr();

  /**
   * Assignment of raw pointer.
   */
  unique_ptr<T>& operator=(T* o);

  /*
   * Equality operators.
   */
  bool operator==(const unique_ptr<T>& o) {
    return get() == o.get();
  }
  bool operator!=(const unique_ptr<T>& o) {
    return get() != o.get();
  }

  /*
   * Inequality operators.
   */
  bool operator<(const unique_ptr<T>& o) {
    return get() < o.get();
  }
  bool operator>(const unique_ptr<T>& o) {
    return get() > o.get();
  }
  bool operator<=(const unique_ptr<T>& o) {
    return get() <= o.get();
  }
  bool operator>=(const unique_ptr<T>& o) {
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

  /**
   * Release the raw pointer.
   */
  T* release();

private:
  /**
   * Raw pointer.
   */
  std::unique_ptr<T> ptr;
};
}

template<class T>
bi::unique_ptr<T>::unique_ptr() :
    ptr(nullptr) {
  //
}

template<class T>
bi::unique_ptr<T>::unique_ptr(T* ptr) :
    ptr(ptr) {
  //
}

template<class T>
bi::unique_ptr<T>::unique_ptr(const unique_ptr<T>& ptr) :
    ptr(ptr.ptr) {
  //
}

template<class T>
bi::unique_ptr<T>::~unique_ptr() {
  //
}

template<class T>
inline bi::unique_ptr<T>& bi::unique_ptr<T>::operator=(T* ptr) {
  if (this->ptr.get() != ptr) {
    this->ptr.reset(ptr);
  }
  return *this;
}

template<class T>
inline bi::unique_ptr<T>::operator bool() const {
  return ptr.get() != nullptr;
}

template<class T>
inline T& bi::unique_ptr<T>::operator*() const {
  return *ptr;
}

template<class T>
inline T* bi::unique_ptr<T>::operator->() const {
  return ptr.get();
}

template<class T>
inline T* bi::unique_ptr<T>::get() const {
  return ptr.get();
}

template<class T>
inline T* bi::unique_ptr<T>::release() {
  return ptr.release();
}
