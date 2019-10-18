/**
 * @file
 */
#pragma once

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
class InitPtr {
  template<class U> friend class SharedPtr;
  template<class U> friend class WeakPtr;
  template<class U> friend class InitPtr;
public:
  using value_type = T;

  /**
   * Constructor.
   */
  explicit InitPtr(T* ptr = nullptr) :
      ptr(ptr) {
    //
  }

  /**
   * Destructor.
   */
  ~InitPtr() {
    release();
  }

  /**
   * Get the raw pointer.
   */
  T* get() const {
    return ptr;
  }

  /**
   * Get the raw pointer as const.
   */
  T* pull() const {
    return ptr;
  }

  /**
   * Replace.
   */
  void replace(T* ptr) {
    this->ptr = ptr;
  }

  /**
   * Release.
   */
  void release() {
    ptr = nullptr;
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

  /**
   * Equal comparison.
   */
  bool operator==(const SharedPtr<T>& o) const {
    return ptr == o.ptr;
  }
  bool operator==(const WeakPtr<T>& o) const {
    return ptr == o.ptr;
  }
  bool operator==(const InitPtr<T>& o) const {
    return ptr == o.ptr;
  }
  bool operator==(const T* o) const {
    return ptr == o;
  }

  /**
   * Not equal comparison.
   */
  bool operator!=(const SharedPtr<T>& o) const {
    return ptr != o.ptr;
  }
  bool operator!=(const WeakPtr<T>& o) const {
    return ptr != o.ptr;
  }
  bool operator!=(const InitPtr<T>& o) const {
    return ptr != o.ptr;
  }
  bool operator!=(const T* o) const {
    return ptr != o;
  }

  /**
   * Is the pointer not null?
   */
  operator bool() const {
    return ptr != nullptr;
  }

  /**
   * Dynamic cast.
   */
  template<class U>
  auto dynamic_pointer_cast() const {
    U cast;
    cast.replace(dynamic_cast<typename U::value_type*>(ptr));
    return cast;
  }

  /**
   * Static cast.
   */
  template<class U>
  auto static_pointer_cast() const {
    U cast;
    cast.replace(static_cast<typename U::value_type*>(ptr));
    return cast;
  }

private:
  /**
   * Raw pointer.
   */
  T* ptr;
};
}
