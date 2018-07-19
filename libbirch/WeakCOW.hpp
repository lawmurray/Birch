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
  template<class U> friend class WeakCOW;
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
  WeakCOW(T* object, World* world, World* current) :
      super_type(object, world, current) {
    //
  }

  /**
   * Copy constructor.
   */
  WeakCOW(const WeakCOW<T>& o) :
      super_type(o) {
    //
  }

  /**
   * Copy constructor.
   */
  WeakCOW(const SharedCOW<T>& o) :
      super_type(o) {
    //
  }

  /**
   * Copy constructor.
   */
  template<class U>
  WeakCOW(const Optional<SharedCOW<U>>& o) :
      super_type(o) {
    //
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
  template<class U> friend class WeakCOW;
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

  WeakCOW(const SharedCOW<Any>& o) :
      object(o.object),
      world(o.world),
      current(o.current) {
    //
  }

  template<class U>
  WeakCOW(const Optional<SharedCOW<U>>& o) :
      WeakCOW(o.query() ? o.get() : nullptr) {
    //
  }

  WeakCOW(WeakCOW<Any> && o) = default;

  WeakCOW<Any>& operator=(const WeakCOW<Any>& o) {
    bi_assert_msg(world->hasLaunchAncestor(o.world),
        "when a fiber yields an object, that object cannot be kept by the caller");
    auto old = std::move(object);
    // ^ ensures next assignment doesn't destroy o

    object = o.pull();
    current = o.current;
    return *this;
  }

  WeakCOW<Any>& operator=(WeakCOW<Any> && o) = default;

  Any* pull() const {
    auto shared = object.lock();
    if (shared) {
      shared = world->getNoCopy(shared, current);
    }
    auto self = const_cast<WeakCOW<Any>*>(this);
    self->object = shared;
    self->current = world;
    return shared.get();
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
