/**
 * @file
 */
#pragma once

#include "libbirch/SharedCOW.hpp"
#include "libbirch/WeakPtr.hpp"
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
class WeakCOW: public WeakCOW<typename super_type<T>::type> {
  template<class U> friend class SharedCOW;
public:
  using value_type = T;
  using this_type = WeakCOW<T>;
  using super_type = WeakCOW<typename super_type<T>::type>;
  using root_type = typename super_type::root_type;

  /**
   * Constructor.
   */
  WeakCOW(T* object = nullptr) :
      super_type(object) {
    //
  }

  /**
   * Constructor.
   */
  WeakCOW(const Nil& object) :
      super_type(object) {
    //
  }

  /**
   * Constructor.
   */
  WeakCOW(const SharedPtr<T>& object) :
      super_type(object) {
    //
  }

  /**
   * Constructor.
   */
  WeakCOW(const WeakPtr<T>& object) :
      super_type(object) {
    //
  }

  /**
   * Constructor.
   */
  WeakCOW(const SharedCOW<T>& o) :
      super_type(o) {
    //
  }

  /**
   * Constructor.
   */
  WeakCOW(T* object, World* world, World* current) :
      super_type(object, world, current) {
    //
  }

  /**
   * Copy constructor.
   */
  WeakCOW(const WeakCOW<T>& o) = default;

  /**
   * Move constructor.
   */
  WeakCOW(WeakCOW<T>&& o) = default;

  /**
   * Copy assignment.
   */
  WeakCOW<T>& operator=(const WeakCOW<T>& o) = default;

  /**
   * Move assignment.
   */
  WeakCOW<T>& operator=(WeakCOW<T>&& o) = default;

  /**
   * Generic copy assignment.
   */
  template<class U>
  WeakCOW<T>& operator=(const WeakCOW<U>& o) {
    root_type::operator=(o);
    return *this;
  }

  /**
   * Generic move assignment.
   */
  template<class U>
  WeakCOW<T>& operator=(WeakCOW<U>&& o) {
    root_type::operator=(o);
    return *this;
  }

  /**
   * Pull through generations.
   */
  T* pull() const {
    return static_cast<T*>(root_type::pull());
  }
};

template<>
class WeakCOW<Any> {
  template<class U> friend class SharedCOW;
public:
  using value_type = Any;
  using this_type = WeakCOW<value_type>;
  using root_type = this_type;

  WeakCOW(Any* object = nullptr) :
      object(object),
      world(fiberWorld),
      current(fiberWorld) {
    //
  }

  WeakCOW(const Nil& object) :
      world(fiberWorld),
      current(fiberWorld) {
    //
  }

  WeakCOW(const SharedPtr<Any>& object) :
      object(object),
      world(fiberWorld),
      current(fiberWorld) {
    //
  }

  WeakCOW(const WeakPtr<Any>& object) :
      object(object),
      world(fiberWorld),
      current(fiberWorld) {
    //
  }

  WeakCOW(const SharedCOW<Any>& o) :
      object(o.object),
      world(o.world),
      current(o.current) {
    //
  }

  WeakCOW(Any* object, World* world, World* current) :
      object(object),
      world(world),
      current(current) {
    //
  }

  WeakCOW(const WeakCOW<Any>& o) :
      object(o.object),
      world(fiberClone ? fiberWorld : o.world),
      current(o.current) {
    //
  }

  WeakCOW(WeakCOW<Any> && o) = default;

  WeakCOW<Any>& operator=(const WeakCOW<Any>& o) {
    auto old = std::move(object);  // ^ ensures next assign won't destroy o
    object = o.pull();
    current = o.current;
    return *this;
  }

  WeakCOW<Any>& operator=(WeakCOW<Any>&& o) = default;

  Any* pull() const {
    if (object) {
      auto self = const_cast<WeakCOW<Any>*>(this);
      self->object = self->world->getNoCopy(object.get(), current);
      self->current = self->world;
    }
    return object.get();
  }

protected:
  /**
   * The object.
   */
  WeakPtr<Any> object;

  /**
   * The world to which the object should belong (although it may belong to
   * a clone ancestor of this world).
   */
  World* world;

  /**
   * Current world.
   */
  World* current;
};
}
