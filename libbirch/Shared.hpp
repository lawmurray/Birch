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
      ptr(ptr) {
    if (ptr) {
      ptr->incShared();
    }
  }

  /**
   * Copy constructor.
   */
  Shared(const Shared& o) {
    auto ptr = o.ptr.load();
    if (ptr) {
      ptr->incShared();
    }
    this->ptr.set(ptr);
  }

  /**
   * Generic shared pointer copy constructor.
   */
  template<class U, std::enable_if_t<std::is_base_of<T,U>::value,int> = 0>
  Shared(const Shared<U>& o) {
    auto ptr = o.ptr.load();
    if (ptr) {
      ptr->incShared();
    }
    this->ptr.set(ptr);
  }

  /**
   * Generic other pointer copy constructor.
   */
  template<class Q, std::enable_if_t<std::is_base_of<T,typename Q::value_type>::value,int> = 0>
  Shared(const Q& o) {
    auto ptr = o.ptr.load();
    if (ptr) {
      ptr->incShared();
    }
    this->ptr.set(ptr);
  }

  /**
   * Move constructor.
   */
  Shared(Shared&& o) {
    ptr.set(o.ptr.exchange(nullptr));
  }

  /**
   * Generic move constructor.
   */
  template<class U, std::enable_if_t<std::is_base_of<T,U>::value,int> = 0>
  Shared(Shared<U>&& o) {
    ptr.set(o.ptr.exchange(nullptr));
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
    auto ptr = this->ptr.get();
    if (ptr) {
      ptr->incShared();
    }
  }

  /**
   * Copy assignment.
   */
  Shared& operator=(const Shared& o) {
    replace(o.get());
    return *this;
  }

  /**
   * Generic shared pointer copy assignment.
   */
  template<class U, std::enable_if_t<std::is_base_of<T,U>::value,int> = 0>
  Shared& operator=(const Shared<U>& o) {
    replace(o.get());
    return *this;
  }

  /**
   * Generic other pointer copy assignment.
   */
  template<class Q, std::enable_if_t<std::is_base_of<T,typename Q::value_type>::value,int> = 0>
  Shared& operator=(const Q& o) {
    replace(o.get());
    return *this;
  }

  /**
   * Move assignment.
   */
  Shared& operator=(Shared&& o) {
    auto ptr = o.ptr.exchange(nullptr);
    auto old = this->ptr.exchange(ptr);
    if (old) {
      old->decShared();
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
    if (old) {
      old->decShared();
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
   * Replace.
   */
  void replace(T* ptr) {
    if (ptr) {
      ptr->incShared();
    }
    auto old = this->ptr.exchange(ptr);
    if (old) {
      old->decShared();
    }
  }

  /**
   * Release.
   */
  void release() {
    auto old = ptr.exchange(nullptr);
    if (old) {
      old->decShared();
    }
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
   * Mark.
   */
  void mark() {
    auto o = ptr.load();
    o->breakShared();  // break the reference
    if (o) {
      o->Any::mark();
    }
  }

  /**
   * Scan.
   */
  void scan(const bool reachable) {
    auto o = ptr.load();
    if (reachable) {
      o->incShared();  // restore the broken reference
    }
    if (o) {
      o->Any::scan(reachable);
    }
  }

  /**
   * Collect.
   */
  void collect() {
    auto o = ptr.exchange(nullptr);  // reference still broken, just set null
    if (o) {
      o->Any::collect();
    }
  }

private:
  /**
   * Raw pointer.
   */
  Atomic<T*> ptr;
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
