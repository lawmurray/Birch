/**
 * @file
 */
#pragma once

namespace bi {
/**
 * Smart pointer reflecting ownership of the object associated with a raw
 * pointer. Unlike other smart pointers, allows assignment to a raw
 * pointer.
 *
 * @tparam T Type.
 */
template<class T>
class owned_ptr {
public:
  /**
   * Constructor. The pointer is initialised to null.
   */
  owned_ptr();

  /**
   * Constructor.
   */
  explicit owned_ptr(T* ptr);

  /**
   * Destructor.
   */
  ~owned_ptr();

  /*
   * Delete default copy constructor and assignment operator, as no more than one
   * owner allowed.
   */
  owned_ptr(const owned_ptr<T>& o) = delete;
  owned_ptr<T>& operator=(const owned_ptr<T>& o) = delete;

  /**
   * Assignment of raw pointer.
   */
  owned_ptr<T>& operator=(T* o);

  /*
   * Equality operators.
   */
  bool operator==(const owned_ptr<T>& o) {
    return get() == o.get();
  }
  bool operator!=(const owned_ptr<T>& o) {
    return get() != o.get();
  }

  /*
   * Inequality operators.
   */
  bool operator<(const owned_ptr<T>& o) {
    return get() < o.get();
  }
  bool operator>(const owned_ptr<T>& o) {
    return get() > o.get();
  }
  bool operator<=(const owned_ptr<T>& o) {
    return get() <= o.get();
  }
  bool operator>=(const owned_ptr<T>& o) {
    return get() >= o.get();
  }

  /**
   * Cast to raw pointer.
   */
  operator T*() const;

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
  T* ptr;
};
}

template<class T>
bi::owned_ptr<T>::owned_ptr() : ptr(nullptr) {
  //
}

template<class T>
bi::owned_ptr<T>::owned_ptr(T* ptr) : ptr(ptr) {
  //
}

template<class T>
bi::owned_ptr<T>::~owned_ptr() {
  delete ptr;
}

template<class T>
inline bi::owned_ptr<T>& bi::owned_ptr<T>::operator=(T* ptr) {
  if (this->ptr != ptr) {
    delete this->ptr;
    this->ptr = ptr;
  }
  return *this;
}

template<class T>
inline bi::owned_ptr<T>::operator T*() const {
  return ptr;
}

template<class T>
inline bi::owned_ptr<T>::operator bool() const {
  return ptr != nullptr;
}

template<class T>
inline T& bi::owned_ptr<T>::operator*() const {
  return *ptr;
}

template<class T>
inline T* bi::owned_ptr<T>::operator->() const {
  return ptr;
}

template<class T>
inline T* bi::owned_ptr<T>::get() const {
  return ptr;
}
