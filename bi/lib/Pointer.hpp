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
class Pointer {
  template<class U>
  friend class Pointer;
public:
  /**
   * Constructor.
   */
  Pointer(T* ptr = nullptr) : ptr(ptr), page(-1) {
    //
  }

  /**
   * Generic constructor.
   */
  template<class U>
  Pointer(U* ptr = nullptr) : ptr(ptr), page(-1) {
    //
  }

  /**
   * Copy constructor.
   */
  Pointer(const Pointer<T>& o) : ptr(o.ptr), page(o.page) {
    //
  }

  /**
   * Generic copy constructor.
   */
  template<class U>
  explicit Pointer(const Pointer<U>& o) : ptr(o.ptr), page(o.page) {
    //
  }

  /**
   * Destructor.
   */
  ~Pointer() {
    //
  }

  /*
   * Equality operators.
   */
  bool operator==(const Pointer<T>& o) const {
    return ptr == o.ptr && page == o.page;
  }
  bool operator!=(const Pointer<T>& o) const {
    return !(*this == o);
  }

  /*
   * Inequality operators.
   */
  bool operator<(const Pointer<T>& o) const {
    return ptr < o.ptr;
  }
  bool operator>(const Pointer<T>& o) const {
    return ptr > o.ptr;
  }
  bool operator<=(const Pointer<T>& o) const {
    return ptr <= o.ptr;
  }
  bool operator>=(const Pointer<T>& o) const {
    return ptr >= o.ptr;
  }

  /*
   * Arithmetic operators.
   */
  Pointer<T>& operator+=(const ptrdiff_t o) {
    ptr += o;
    return *this;
  }
  Pointer<T>& operator-=(const ptrdiff_t o) {
    ptr -= o;
    return *this;
  }
  Pointer<T> operator+(const ptrdiff_t o) const {
    Pointer<T> result(*this);
    result += o;
    return result;
  }
  Pointer<T> operator-(const ptrdiff_t o) const {
    Pointer<T> result(*this);
    result -= o;
    return result;
  }

  /**
   * Cast to raw pointer.
   */
  operator T*() {
    if (page < 0) {
      return ptr;
    } else {
      assert(false);
    }
  }
  operator T* const() const {
    if (page < 0) {
      return ptr;
    } else {
      assert(false);
    }
  }

  /**
   * Cast to bool (check for null pointer).
   */
  operator bool() const {
    return ptr != nullptr;
  }

  /**
   * Pointer casts.
   */
  template<class U>
  operator Pointer<U>() {
    return Pointer<U>(*this);
  }
  template<class U>
  operator Pointer<U>() const {
    return Pointer<U>(*this);
  }

  /**
   * Other casts (defer to value type).
   */
  template<class U>
  operator U() {
    return *ptr;
  }
  template<class U>
  operator U() const {
    return *ptr;
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

  /**
   * Heap page index.
   */
  int page;
};
}
