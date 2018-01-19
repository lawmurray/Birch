/**
 * @file
 */
#pragma once

#include "libbirch/SharedPointer.hpp"
#include "libbirch/Optional.hpp"

namespace bi {
/**
 * Shared pointer with copy-on-write semantics.
 *
 * @ingroup libbirch
 *
 * @tparam T Type.
 */
template<class T>
class WeakPointer: public WeakPointer<typename super_type<T>::type> {
public:
  using value_type = T;
  using this_type = WeakPointer<T>;
  using super_type = WeakPointer<typename super_type<T>::type>;
  using root_type = typename super_type::root_type;

  /**
   * Default constructor.
   */
  WeakPointer(const std::nullptr_t& o = nullptr) :
      super_type(o) {
    //
  }

  /**
   * Generic copy constructor.
   */
  template<class U>
  WeakPointer(const WeakPointer<U>& o) :
      super_type(o) {
    //
  }

  /**
   * Generic copy constructor from shared pointer.
   */
  template<class U>
  WeakPointer(const SharedPointer<U>& o) :
      super_type(o) {
    //
  }

  /**
   * Generic copy constructor from optional shared pointer.
   */
  template<class U>
  WeakPointer(const Optional<SharedPointer<U>>& o) :
      super_type(o) {
    //
  }

  /**
   * Lock the pointer.
   */
  SharedPointer<T> lock() const {
#ifndef NDEBUG
    return root_type::lock().template dynamic_pointer_cast<T>();
#else
    return root_type::lock().template static_pointer_cast<T>();
#endif
  }
};

template<>
class WeakPointer<Any> {
  friend class SharedPointer<Any> ;
  friend class WeakPointer<const Any> ;
public:
  using value_type = Any;
  using this_type = WeakPointer<value_type>;
  using root_type = this_type;

  WeakPointer(const std::nullptr_t& o = nullptr) {
    //
  }

  template<class U>
  WeakPointer(const WeakPointer<U>& o) :
      ptr(o.ptr) {
    //
  }

  template<class U>
  WeakPointer(const SharedPointer<U>& o) :
      ptr(o.ptr) {
    //
  }

  template<class U>
  WeakPointer(const Optional<SharedPointer<U>>& o) {
    if (o.query()) {
      *this = o.get();
    }
  }

  SharedPointer<Any> lock() const {
    return ptr.lock();
  }

protected:
  /**
   * Wrapped smart pointer.
   */
  std::weak_ptr<Any> ptr;
};

template<>
class WeakPointer<const Any> {
  friend class SharedPointer<const Any> ;
public:
  using value_type = const Any;
  using this_type = WeakPointer<const Any>;
  using root_type = this_type;

  WeakPointer(const std::nullptr_t& o = nullptr) {
    //
  }

  template<class U>
  WeakPointer(const WeakPointer<U>& o) :
      ptr(o.ptr) {
    //
  }

  template<class U>
  WeakPointer(const SharedPointer<U>& o) :
      ptr(o.ptr) {
    //
  }

  template<class U>
  WeakPointer(const Optional<SharedPointer<U>>& o) {
    if (o.query()) {
      *this = o.get();
    }
  }

  SharedPointer<const Any> lock() const {
    return SharedPointer<const Any>(ptr.lock());
  }

protected:
  /**
   * Wrapped smart pointer.
   */
  std::weak_ptr<const Any> ptr;
};
}
