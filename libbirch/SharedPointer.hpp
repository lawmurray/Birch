/**
 * @file
 */
#pragma once

#include "libbirch/global.hpp"
#include "libbirch/World.hpp"
#include "libbirch/Any.hpp"

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
  SharedPointer() :
      SharedPointer(std::make_shared<T>()) {
    //
  }

  /**
   * Null constructor.
   */
  SharedPointer(const std::nullptr_t& o) :
      super_type(o) {
    //
  }

  /**
   * Generic constructor from STL shared pointer.
   */
  SharedPointer(const std::shared_ptr<T>& ptr) :
      super_type(ptr) {

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
   * Value assignment.
   */
  template<class U,
      typename = std::enable_if_t<bi::has_assignment<T,U>::value>>
  SharedPointer<T>& operator=(const U& o) {
    *get() = o;
    return *this;
  }

  /**
   * Value conversion.
   */
  template<class U,
      typename = std::enable_if_t<bi::has_conversion<T,U>::value>>
  operator U() {
    return static_cast<U>(*get());
  }
  template<class U,
      typename = std::enable_if_t<bi::has_conversion<T,U>::value>>
  operator U() const {
    return static_cast<U>(*get());
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
  const T* get() const {
#ifndef NDEBUG
    return dynamic_cast<const T*>(root_type::get());
#else
    return static_cast<const T*>(root_type::get());
#endif
  }

  /**
   * Dereference.
   */
  T& operator*() {
    return *get();
  }
  const T& operator*() const {
    return *get();
  }

  /**
   * Member access.
   */
  T* operator->() {
    return get();
  }
  const T* operator->() const {
    return get();
  }

  /**
   * Call operator.
   */
  template<class ... Args>
  auto operator()(Args ... args) {
    return (*get())(args...);
  }
  template<class ... Args>
  auto operator()(Args ... args) const {
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

  SharedPointer() :
      SharedPointer(std::make_shared<Any>()) {
    //
  }

  SharedPointer(const std::nullptr_t& o) :
      ptr(o) {
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

  /**
   * Is the pointer not null?
   */
  bool query() const {
    return static_cast<bool>(ptr);
  }

  Any* get() {
    /* copy-on-write, and update the pointer for next time */
    ptr = fiberWorld->get(ptr);
    return ptr.get();
  }
  const Any* get() const {
    /* given the way C++ code is generated, a const SharedPointer<Any> only
     * occurs for member variables when a read-only function or fiber is
     * called; so treat this as though it were a SharedPointer<const Any> */
    return ptr.get();
  }

  Any& operator*() {
    return *get();
  }
  const Any& operator*() const {
    return *get();
  }

  Any* operator->() {
    return get();
  }
  const Any* operator->() const {
    return get();
  }

  /**
   * Dynamic cast. Returns `nullptr` if the cast if unsuccessful.
   */
  template<class U>
  SharedPointer<U> dynamic_pointer_cast() {
    return SharedPointer<U>(std::dynamic_pointer_cast<U>(ptr));
  }
  template<class U>
  SharedPointer<const U> dynamic_pointer_cast() const {
    return SharedPointer<const U>(std::dynamic_pointer_cast<const U>(ptr));
  }

  /**
   * Static cast. Undefined if unsuccessful.
   */
  template<class U>
  SharedPointer<U> static_pointer_cast() {
    return SharedPointer<U>(std::static_pointer_cast<U>(ptr));
  }
  template<class U>
  SharedPointer<U> static_pointer_cast() const {
    return SharedPointer<const U>(std::static_pointer_cast<const U>(ptr));
  }

protected:
  /**
   * Wrapped smart pointer.
   */
  std::shared_ptr<Any> ptr;
};

template<>
class SharedPointer<const Any> {
  template<class U> friend class SharedPointer;
  template<class U> friend class WeakPointer;
public:
  using value_type = const Any;
  using this_type = SharedPointer<const Any>;
  using root_type = this_type;

  SharedPointer() :
      SharedPointer(std::make_shared<const Any>()) {
    //
  }

  SharedPointer(const std::nullptr_t& o) :
      ptr(o) {
    //
  }

  template<class U>
  SharedPointer(const std::shared_ptr<U>& ptr) :
      ptr(ptr) {
    //
  }

  template<class U>
  SharedPointer(const SharedPointer<U>& o) :
      ptr(o.ptr) {
    //
  }

  bool query() const {
    return static_cast<bool>(ptr);
  }

  const Any* get() {
    /* read-only, so no need for copy-on-write */
    return ptr.get();
  }
  const Any* get() const {
    /* read-only, so no need for copy-on-write */
    return ptr.get();
  }

  const Any& operator*() {
    return *get();
  }
  const Any& operator*() const {
    return *get();
  }

  const Any* operator->() {
    return get();
  }
  const Any* operator->() const {
    return get();
  }

  template<class U>
  SharedPointer<const U> dynamic_pointer_cast() const {
    return SharedPointer<const U>(std::dynamic_pointer_cast<const U>(ptr));
  }

  template<class U>
  SharedPointer<const U> static_pointer_cast() const {
    return SharedPointer<const U>(std::static_pointer_cast<const U>(ptr));
  }

protected:
  /**
   * Wrapped smart pointer.
   */
  std::shared_ptr<const Any> ptr;
};
}
