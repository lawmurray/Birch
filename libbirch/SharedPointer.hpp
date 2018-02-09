/**
 * @file
 */
#pragma once

#include "libbirch/global.hpp"
#include "libbirch/World.hpp"
#include "libbirch/Any.hpp"
#include "libbirch/Wrapper.hpp"

namespace bi {
/**
 * Shared pointer with copy-on-write semantics.
 *
 * @ingroup libbirch
 *
 * @tparam T Type.
 */
template<class T>
class SharedPointer: public SharedPointer<typename super_type<T>::type> {
  template<class U> friend class SharedPointer;
  template<class U> friend class WeakPointer;
public:
  using value_type = T;
  using this_type = SharedPointer<T>;
  using super_type = SharedPointer<typename super_type<T>::type>;
  using root_type = typename super_type::root_type;

  /**
   * Default constructor.
   */
  SharedPointer(const std::nullptr_t& o = nullptr) {
    //
  }

  /**
   * Constructor.
   */
  SharedPointer(const std::shared_ptr<T>& ptr) :
      super_type(ptr) {
    //
  }

  /**
   * Generic constructor.
   */
  template<class U>
  SharedPointer(const SharedPointer<U>& o) :
      super_type(o) {
    //
  }

  /**
   * Generic constructor.
   */
  template<class U>
  SharedPointer(const WeakPointer<U>& o) :
      super_type(o) {
    //
  }

  /**
   * Value assignment.
   */
  template<class U,
      typename = std::enable_if_t<bi::has_assignment<T,U>::value>>
  SharedPointer<T>& operator=(const U& o) {
    *Wrapper<T>(get()) = o;
    return *this;
  }

  /**
   * Value conversion.
   */
  template<class U,
      typename = std::enable_if_t<bi::has_conversion<T,U>::value>>
  operator U() {
    return static_cast<U>(*Wrapper<T>(get()));
  }

  /**
   * Get the raw pointer.
   */
  T* get() {
#ifndef NDEBUG
    return dynamic_cast<T*>(root_type::get());
#else
    return static_cast<T*>(root_type::get());
#endif
  }

  /**
   * Dereference.
   */
  T& operator*() {
    return *get();
  }

  /**
   * Member access.
   */
  Wrapper<T> operator->() {
    return Wrapper<T>(get());
  }

  /**
   * Call operator.
   */
  template<class ... Args>
  auto operator()(Args ... args) {
    return (*get())(args...);
  }
};

template<>
class SharedPointer<Any> {
  template<class U> friend class SharedPointer;
  template<class U> friend class WeakPointer;
public:
  using value_type = Any;
  using this_type = SharedPointer<value_type>;
  using root_type = this_type;

  SharedPointer(const std::nullptr_t& o = nullptr) {
    //
  }

  SharedPointer(const std::shared_ptr<Any>& ptr) :
      ptr(ptr) {
    //
  }

  template<class U>
  SharedPointer(const SharedPointer<U>& o) :
      ptr(o.ptr) {
    //
  }

  template<class U>
  SharedPointer(const WeakPointer<U>& o) :
      ptr(o.ptr.lock()) {
    //
  }

  SharedPointer<Any>& operator=(const SharedPointer<Any>& o) {
    assert(!o.ptr || o.ptr->getWorld() == fiberWorld);
    ptr = o.ptr;
    return *this;
  }

  /**
   * Is the pointer not null?
   */
  bool query() const {
    return static_cast<bool>(ptr);
  }

  Any* get() {
    ptr = fiberWorld->get(ptr);
    return ptr.get();
  }

  Any& operator*() {
    return *get();
  }

  Any* operator->() {
    return get();
  }

  /**
   * Dynamic cast. Returns `nullptr` if the cast if unsuccessful.
   */
  template<class U>
  SharedPointer<U> dynamic_pointer_cast() {
    return SharedPointer<U>(std::dynamic_pointer_cast < U > (ptr));
  }

  /**
   * Static cast. Undefined if unsuccessful.
   */
  template<class U>
  SharedPointer<U> static_pointer_cast() {
    return SharedPointer<U>(std::static_pointer_cast < U > (ptr));
  }

protected:
  /**
   * Shared pointer to the object.
   */
  std::shared_ptr<Any> ptr;
};
}
