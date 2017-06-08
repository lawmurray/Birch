/**
 * @file
 */
#pragma once

namespace bi {
/**
 * Relocatable pointer.
 *
 * @ingroup library
 *
 * @tparam T Type.
 */
template<class T>
class reloc_ptr {
  template<class U>
  friend class reloc_ptr;
public:
  /**
   * Constructor.
   */
  reloc_ptr(T* ptr = nullptr) : ptr(ptr) {
    //
  }

  /**
   * Generic constructor.
   */
  template<class U>
  reloc_ptr(U* ptr = nullptr) : ptr(ptr) {
    //
  }

  /**
   * Copy constructor.
   */
  reloc_ptr(const reloc_ptr<T>& o) : ptr(o.ptr) {
    //
  }

  /**
   * Generic copy constructor.
   */
  template<class U>
  reloc_ptr(const reloc_ptr<U>& o) : ptr(o.ptr) {
    //
  }

  /**
   * Destructor.
   */
  ~reloc_ptr() {
    //
  }

  /*
   * Equality operators.
   */
  bool operator==(const reloc_ptr<T>& o) const {
    return ptr == o.ptr;
  }
  bool operator!=(const reloc_ptr<T>& o) const {
    return ptr != o.ptr;
  }

  /*
   * Inequality operators.
   */
  bool operator<(const reloc_ptr<T>& o) const {
    return ptr < o.ptr;
  }
  bool operator>(const reloc_ptr<T>& o) const {
    return ptr > o.ptr;
  }
  bool operator<=(const reloc_ptr<T>& o) const {
    return ptr <= o.ptr;
  }
  bool operator>=(const reloc_ptr<T>& o) const {
    return ptr >= o.ptr;
  }

  /*
   * Arithmetic operators.
   */
  reloc_ptr<T>& operator+=(const ptrdiff_t o) {
    ptr += o;
    return *this;
  }
  reloc_ptr<T>& operator-=(const ptrdiff_t o) {
    ptr -= o;
    return *this;
  }
  reloc_ptr<T> operator+(const ptrdiff_t o) const {
    reloc_ptr<T> result(*this);
    result += o;
    return result;
  }
  reloc_ptr<T> operator-(const ptrdiff_t o) const {
    reloc_ptr<T> result(*this);
    result -= o;
    return result;
  }

  /**
   * Cast to raw pointer.
   */
  operator T*() {
    return ptr;
  }
  operator T* const() const {
    return ptr;
  }

  /**
   * Cast to bool (check for null pointer).
   */
  operator bool() const {
    return ptr != nullptr;
  }

  /**
   * Dereference.
   */
  T& operator*() const {
    return *ptr;
  }

  /**
   * Member access.
   */
  T* operator->() const {
    return ptr;
  }

private:
  /**
   * Raw pointer.
   */
  T* ptr;
};
}
