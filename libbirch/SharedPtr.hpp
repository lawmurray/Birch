/**
 * @file
 */
#pragma once

#include "libbirch/class.hpp"
#include "libbirch/Counted.hpp"

namespace libbirch {
/**
 * Shared pointer with intrusive implementation.
 *
 * @ingroup libbirch
 *
 * @tparam T Type, must derive from Counted.
 */
template<class T>
class SharedPtr: public SharedPtr<typename bi::type::super_type<T>::type> {
public:
  using value_type = T;
  using super_type = SharedPtr<typename bi::type::super_type<value_type>::type>;
  using this_type = SharedPtr<value_type>;

  /**
   * Constructor.
   */
  explicit SharedPtr(value_type* ptr = nullptr) :
      super_type(ptr) {
    //
  }

  /**
   * Constructor.
   */
  template<class Q, std::enable_if_t<!std::is_pointer<Q>::value && is_base_of<this_type,Q>::value,int> = 0>
  SharedPtr(const Q& o) :
      super_type(o) {
    //
  }

  /**
   * Get the raw pointer.
   */
  T* get() const {
    return static_cast<T*>(super_type::get());
  }

  /**
   * Get the raw pointer as const.
   */
  T* pull() const {
    return static_cast<T*>(super_type::pull());
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
};

/**
 * Shared pointer with intrusive implementation.
 *
 * @ingroup libbirch
 */
template<>
class SharedPtr<Counted> {
public:
  using value_type = Counted;
  using this_type = SharedPtr<Counted>;

  /**
   * Constructor.
   */
  explicit SharedPtr(value_type* ptr = nullptr) :
      ptr(ptr) {
    if (ptr) {
      ptr->incShared();
    }
  }

  /**
   * Constructor.
   */
  template<class Q, std::enable_if_t<!std::is_pointer<Q>::value && is_base_of<this_type,Q>::value,int> = 0>
  SharedPtr(const Q& o) :
      ptr(o.get()) {
    if (ptr) {
      ptr->incShared();
    }
  }

  /**
   * Copy constructor.
   */
  SharedPtr(const SharedPtr& o) {
    if (ptr) {
      assert(ptr->numShared() > 0);
      ptr->incShared();
    }
  }

  /**
   * Move constructor.
   */
  SharedPtr(SharedPtr&& o) {
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
    if (o.ptr) {
      o.ptr->incShared();
    }
    auto old = ptr;
    ptr = o.ptr;
    if (old) {
      old->decShared();
    }
    return *this;
  }

  /**
   * Move assignment.
   */
  SharedPtr& operator=(SharedPtr&& o) {
    auto old = ptr;
    ptr = o.ptr;
    o.ptr = nullptr;
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
    return ptr != nullptr;
  }

  /**
   * Get the raw pointer.
   */
  Counted* get() const {
    assert(!ptr || ptr->numShared() > 0);
    return ptr;
  }

  /**
   * Get the raw pointer as const.
   */
  Counted* pull() const {
    assert(!ptr || ptr->numShared() > 0);
    return ptr;
  }

  /**
   * Replace.
   */
  void replace(Counted* ptr) {
    assert(!ptr || ptr->numShared() > 0);
    auto old = this->ptr;
    if (ptr) {
      ptr->incShared();
    }
    this->ptr = ptr;
    if (old) {
      old->decShared();
    }
  }

  /**
   * Release.
   */
  void release() {
    if (ptr) {
      ptr->decShared();
      ptr = nullptr;
    }
  }

  /**
   * Dereference.
   */
  Counted& operator*() const {
    return *get();
  }

  /**
   * Member access.
   */
  Counted* operator->() const {
    return get();
  }

  /**
   * Equal comparison.
   */
  template<class Q>
  bool operator==(const Q& o) const {
    return get() == o.get();
  }

  /**
   * Not equal comparison.
   */
  template<class Q>
  bool operator!=(const Q& o) const {
    return get() != o.get();
  }

private:
  /**
   * Raw pointer.
   */
  Counted* ptr;
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
struct raw_type<SharedPtr<T>> {
  using type = T*;
};
}
