/**
 * @file
 */
#pragma once

#include "libbirch/Any.hpp"
#include "libbirch/type.hpp"

namespace libbirch {
/**
 * Shared pointer with intrusive implementation.
 *
 * @ingroup libbirch
 *
 * @tparam T Type, must derive from Any.
 */
template<class T>
class SharedPtr {
  template<class U> friend class SharedPtr;
  template<class U> friend class WeakPtr;
  template<class U> friend class InitPtr;
public:
  using value_type = T;

  /**
   * Constructor.
   */
  explicit SharedPtr(value_type* ptr = nullptr) :
      ptr(ptr),
      discarded(false) {
    if (ptr) {
      ptr->incShared();
    }
  }

  /**
   * Copy constructor.
   */
  SharedPtr(const SharedPtr& o) :
      ptr(o.ptr),
      discarded(false) {
    if (ptr) {
      ptr->incShared();
    }
  }

  /**
   * Generic copy constructor.
   */
  template<class Q, class U = typename Q::value_type,
      std::enable_if_t<std::is_base_of<T,U>::value,int> = 0>
  SharedPtr(const Q& o) :
      ptr(o.ptr),
      discarded(false) {
    if (ptr) {
      ptr->incShared();
    }
  }

  /**
   * Move constructor.
   */
  SharedPtr(SharedPtr&& o) :
      ptr(o.ptr),
      discarded(false) {
    if (o.isDiscarded()) {
      ptr->restoreShared();
    }
    o.ptr = nullptr;
  }

  /**
   * Generic move constructor.
   */
  template<class U, std::enable_if_t<std::is_base_of<T,U>::value,int> = 0>
  SharedPtr(SharedPtr<U>&& o) :
      ptr(o.ptr),
      discarded(false) {
    if (o.isDiscarded()) {
      ptr->restoreShared();
    }
    o.ptr = nullptr;
  }

  /**
   * Destructor.
   */
  ~SharedPtr() {
    release();
  }

  /**
   * Copy assignment.
   */
  SharedPtr& operator=(const SharedPtr& o) {
    replace(o.ptr);
    return *this;
  }

  /**
   * Generic copy assignment.
   */
  template<class Q, class U = typename Q::value_type,
      std::enable_if_t<std::is_base_of<T,U>::value,int> = 0>
  SharedPtr& operator=(const Q& o) {
    replace(o.ptr);
    return *this;
  }

  /**
   * Move assignment.
   */
  SharedPtr& operator=(SharedPtr&& o) {
    auto old = ptr;
    ptr = o.ptr;
    o.ptr = nullptr;
    if (discarded) {
      if (ptr && !o.isDiscarded()) {
        ptr->discardShared();
      }
      if (old) {
        old->decMemoShared();
      }
    } else {
      if (ptr && o.isDiscarded()) {
        ptr->restoreShared();
      }
      if (old) {
        old->decShared();
      }
    }
    return *this;
  }

  /**
   * Generic move assignment.
   */
  template<class U, std::enable_if_t<std::is_base_of<T,U>::value,int> = 0>
  SharedPtr& operator=(SharedPtr<U>&& o) {
    auto old = ptr;
    ptr = o.ptr;
    o.ptr = nullptr;
    if (discarded) {
      if (ptr && !o.isDiscarded()) {
        ptr->discardShared();
      }
      if (old) {
        old->decMemoShared();
      }
    } else {
      if (ptr && o.isDiscarded()) {
        ptr->restoreShared();
      }
      if (old) {
        old->decShared();
      }
    }
    return *this;
  }

  /**
   * Is the pointer not null?
   *
   * This is used instead of an `operator bool()` so as not to conflict with
   * conversion operators in the referent type.
   */
  bool query() const {
    return ptr != nullptr;
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
    auto old = this->ptr;
    this->ptr = ptr;
    if (discarded) {
      if (ptr) {
        ptr->incMemoShared();
      }
      if (old) {
        old->decMemoShared();
      }
    } else {
      if (ptr) {
        ptr->incShared();
      }
      if (old) {
        old->decShared();
      }
    }
  }

  /**
   * Release.
   */
  void release() {
    if (ptr) {
      if (discarded) {
        ptr->decMemoShared();
      } else {
        ptr->decShared();
      }
      ptr = nullptr;
    }
  }
  
  /**
   * Discard.
   */
  void discard() {
    assert(!discarded);
    if (ptr) {
      discarded = true;
      ptr->discardShared();
    }
  }

  /**
   * Restore.
   */
  void restore() {
    assert(discarded);
    if (ptr) {
      discarded = false;
      ptr->restoreShared();
    }
  }

  /**
   * Has this been discarded?
   */
  bool isDiscarded() const {
    return discarded;
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

private:
  /**
   * Raw pointer.
   */
  T* ptr;

  /**
   * Has the pointer been discarded?
   */
  bool discarded;
};

template<class T>
struct is_value<SharedPtr<T>> {
  static const bool value = false;
};

template<class T>
struct is_pointer<SharedPtr<T>> {
  static const bool value = true;
};

template<class T>
struct raw<SharedPtr<T>> {
  using type = T*;
};
}
