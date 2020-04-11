/**
 * @file
 */
#pragma once

#include "libbirch/Any.hpp"
#include "libbirch/Atomic.hpp"
#include "libbirch/type.hpp"

namespace libbirch {
/**
 * Shared pointer.
 *
 * @ingroup libbirch
 *
 * @tparam T Type, must derive from Any.
 */
template<class T>
class Shared {
  template<class U> friend class Shared;
  template<class U> friend class Weak;
  template<class U> friend class Init;
  template<class U> friend class Lazy;
public:
  using value_type = T;

  /**
   * Constructor.
   */
  explicit Shared(value_type* ptr = nullptr) :
      ptr(ptr),
      discarded(false) {
    if (ptr) {
      ptr->incShared();
    }
  }

  /**
   * Copy constructor.
   */
  Shared(const Shared& o) :
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
  template<class Q, std::enable_if_t<std::is_base_of<T,typename Q::value_type>::value,int> = 0>
  Shared(const Q& o) :
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
  Shared(Shared&& o) :
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
  Shared(Shared<U>&& o) :
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
  ~Shared() {
    release();
  }

  /**
   * Fix after a bitwise copy.
   */
  void bitwiseFix() {
    discarded.set(false);
    if (ptr.get()) {
      ptr.get()->incShared();
    }
  }

  /**
   * Copy assignment.
   */
  Shared& operator=(const Shared& o) {
    replace(o.ptr.load());
    return *this;
  }

  /**
   * Generic copy assignment.
   */
  template<class Q, std::enable_if_t<std::is_base_of<T,typename Q::value_type>::value,int> = 0>
  Shared& operator=(const Q& o) {
    replace(o.ptr.load());
    return *this;
  }

  /**
   * Move assignment.
   */
  Shared& operator=(Shared&& o) {
    auto ptr = o.ptr.exchange(nullptr);
    auto old = this->ptr.exchange(ptr);
    if (discarded.load()) {
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
  Shared& operator=(Shared<U>&& o) {
    auto ptr = o.ptr.exchange(nullptr);
    auto old = this->ptr.exchange(ptr);
    if (discarded.load()) {
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
    auto discarded = this->discarded.load();
    if (ptr) {
      if (discarded) {
        ptr->incMemoShared();
      } else {
        ptr->incShared();
      }
    }
    auto old = this->ptr.exchange(ptr);
    if (old) {
      if (discarded) {
        old->decMemoShared();
      } else {
        old->decShared();
      }
    }
  }

  /**
   * Release.
   */
  void release() {
    auto discarded = this->discarded.load();
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
    auto ptr = this->ptr.load();
    if (ptr && !discarded.exchange(true)) {
      ptr->discardShared();
    }
  }

  /**
   * Restore.
   */
  void restore() {
    auto ptr = this->ptr.load();
    if (ptr && discarded.exchange(false)) {
      ptr->restoreShared();
    }
  }

  /**
   * Has this been discarded?
   */
  bool isDiscarded() const {
    return discarded.load();
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
  Atomic<bool> discarded;
};

template<class T>
struct is_value<Shared<T>> {
  static const bool value = false;
};

template<class T>
struct is_pointer<Shared<T>> {
  static const bool value = true;
};

template<class T>
struct raw<Shared<T>> {
  using type = T*;
};
}
