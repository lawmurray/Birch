/**
 * @file
 */
#pragma once

#include "libbirch/Any.hpp"
#include "libbirch/Atomic.hpp"
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
      discarded(false) {
    auto ptr = o.ptr.load();
    if (ptr) {
      ptr->incShared();
    }
    this->ptr.store(ptr);
  }

  /**
   * Generic copy constructor.
   */
  template<class Q, class U = typename Q::value_type,
      std::enable_if_t<std::is_base_of<T,U>::value,int> = 0>
  SharedPtr(const Q& o) :
      discarded(false) {
    auto ptr = o.ptr.load();
    if (ptr) {
      ptr->incShared();
    }
    this->ptr.store(ptr);
  }

  /**
   * Move constructor.
   */
  SharedPtr(SharedPtr&& o) :
      discarded(false) {
    auto ptr = o.ptr.exchange(nullptr);
    if (ptr && o.isDiscarded()) {
      ptr->restoreShared();
    }
    this->ptr.store(ptr);
  }

  /**
   * Generic move constructor.
   */
  template<class U, std::enable_if_t<std::is_base_of<T,U>::value,int> = 0>
  SharedPtr(SharedPtr<U>&& o) :
      discarded(false) {
    auto ptr = o.ptr.exchange(nullptr);
    if (ptr && o.isDiscarded()) {
      ptr->restoreShared();
    }
    this->ptr.store(ptr);
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
    replace(o.ptr.load());
    return *this;
  }

  /**
   * Generic copy assignment.
   */
  template<class Q, class U = typename Q::value_type,
      std::enable_if_t<std::is_base_of<T,U>::value,int> = 0>
  SharedPtr& operator=(const Q& o) {
    replace(o.ptr.load());
    return *this;
  }

  /**
   * Move assignment.
   */
  SharedPtr& operator=(SharedPtr&& o) {
    auto ptr = o.ptr.exchange(nullptr);
    auto old = this->ptr.exchange(ptr);
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
    auto ptr = o.ptr.exchange(nullptr);
    auto old = this->ptr.exchange(ptr);
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
    return ptr.load() != nullptr;
  }

  /**
   * Get the raw pointer.
   */
  T* get() const {
    return ptr.load();
  }

  /**
   * Get the raw pointer as const.
   */
  T* pull() const {
    return ptr.load();
  }

  /**
   * Replace.
   */
  void replace(T* ptr) {
    auto old = this->ptr.exchange(ptr);
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
    auto old = ptr.exchange(nullptr);
    if (old) {
      if (discarded) {
        old->decMemoShared();
      } else {
        old->decShared();
      }
    }
  }
  
  /**
   * Discard.
   */
  void discard() {
    assert(!discarded);
    auto ptr = this->ptr.load();
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
    auto ptr = this->ptr.load();
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
  Atomic<T*> ptr;

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
