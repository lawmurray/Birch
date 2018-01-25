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
      SharedPointer(std::make_shared<T>(), fiberWorld) {
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
   * Constructor.
   */
  SharedPointer(const std::shared_ptr<T>& ptr,
      const std::shared_ptr<World>& world = fiberWorld) :
      super_type(ptr, world) {
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
   * Copy constructor.
   */
  SharedPointer(const SharedPointer<T>& o) :
      super_type(o) {
    //
  }

  /**
   * Copy assignment.
   */
  SharedPointer<T>& operator=(const SharedPointer<T>& o) {
    super_type::operator=(o);
    return *this;
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
  T* operator->() {
    return get();
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

  SharedPointer() :
      ptr(std::make_shared<Any>()),
      world(fiberWorld) {
    //
  }

  SharedPointer(const std::nullptr_t& o) :
      ptr(o),
      world(fiberWorld) {
    //
  }

  SharedPointer(const std::shared_ptr<Any>& ptr,
      const std::shared_ptr<World>& world = fiberWorld) :
      ptr(ptr),
      world(fiberWorld->isReachable(world.get()) ? fiberWorld : world) {
    //
  }

  template<class U>
  SharedPointer(const SharedPointer<U>& o) :
      ptr(o.ptr),
      world(fiberWorld->isReachable(o.world.get()) ? fiberWorld : o.world) {
    //
  }

  template<class U>
  SharedPointer(const WeakPointer<U>& o) :
      ptr(o.ptr.lock()),
      world(fiberWorld->isReachable(o.world.get()) ? fiberWorld : o.world) {
    //
  }

  SharedPointer(const SharedPointer<Any>& o) :
      ptr(o.ptr),
      world(fiberWorld->isReachable(o.world.get()) ? fiberWorld : o.world) {
    //
  }

  SharedPointer<Any>& operator=(const SharedPointer<Any>& o) {
    ptr = o.ptr;
    world = fiberWorld->isReachable(o.world.get()) ? fiberWorld : o.world;
    return *this;
  }

  /**
   * Is the pointer not null?
   */
  bool query() const {
    return static_cast<bool>(ptr);
  }

  Any* get() {
    ptr = world->get(ptr);
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
    return SharedPointer<U>(std::dynamic_pointer_cast < U > (ptr), world);
  }

  /**
   * Static cast. Undefined if unsuccessful.
   */
  template<class U>
  SharedPointer<U> static_pointer_cast() {
    return SharedPointer<U>(std::static_pointer_cast < U > (ptr), world);
  }

protected:
  /**
   * Shared pointer to the object.
   */
  std::shared_ptr<Any> ptr;

  /**
   * Shared pointer to the world in which the object is required.
   */
  std::shared_ptr<World> world;
};
}
