/**
 * @file
 */
#pragma once

namespace bi {
template<class T> class SharedPtr;
template<class T> class WeakPtr;

/**
 * Shared pointer with intrusive implementation.
 *
 * @ingroup libbirch
 *
 * @tparam T Type, must derive from Counted.
 */
template<class T>
class SharedPtr {
  template<class U> friend class SharedPtr;
  template<class U> friend class WeakPtr;
public:
  /**
   * Constructor.
   */
  SharedPtr(T* ptr = nullptr) :
      ptr(ptr) {
    if (ptr) {
      ptr->incShared();
    }
  }

  /**
   * Copy constructor.
   */
  SharedPtr(const SharedPtr<T>& o) :
      ptr(o.ptr) {
    if (ptr) {
      ptr->incShared();
    }
  }

  /**
   * Move constructor.
   */
  SharedPtr(SharedPtr<T> && o) :
      ptr(o.ptr) {
    o.ptr = nullptr;
  }

  /**
   * Destructor.
   */
  ~SharedPtr() {
    if (ptr) {
      ptr->decShared();
    }
  }

  /**
   * Copy assignment.
   */
  SharedPtr<T>& operator=(const SharedPtr<T>& o) {
    if (ptr != o.ptr) {
      if (ptr) {
        ptr->decShared();
      }
      ptr = o.ptr;
      if (ptr) {
        ptr->incShared();
      }
    }
    return *this;
  }

  /**
   * Move assignment.
   */
  SharedPtr<T>& operator=(SharedPtr<T> && o) {
    if (ptr != o.ptr) {
      ptr = o.ptr;
      o.ptr = nullptr;
    }
    return *this;
  }

  /**
   * Get the raw pointer.
   */
  T* get() const {
    return ptr;
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

  /**
   * Not equal comparison.
   */
  bool operator!=(const SharedPtr<T>& o) const {
    return ptr != o.ptr;
  }

  /**
   * Is the pointer not null?
   */
  operator bool() const {
    return ptr != nullptr;
  }

private:
  /**
   * Raw pointer.
   */
  T* ptr;
};
}
