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
  template<class U> friend class SharedPointer;
  template<class U> friend class WeakPointer;
public:
  using value_type = T;
  using this_type = WeakPointer<T>;
  using super_type = WeakPointer<typename super_type<T>::type>;
  using root_type = typename super_type::root_type;

  /**
   * Constructor.
   */
  WeakPointer(const std::nullptr_t& object = nullptr) :
      super_type(object) {
    //
  }

  /**
   * Constructor.
   */
  WeakPointer(const Nil& object) :
      super_type(object) {
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
};

template<>
class WeakPointer<Any> {
  template<class U> friend class SharedPointer;
  template<class U> friend class WeakPointer;
public:
  using value_type = Any;
  using this_type = WeakPointer<value_type>;
  using root_type = this_type;

  WeakPointer(const std::nullptr_t& object = nullptr) :
      world(fiberWorld) {
    //
  }

  WeakPointer(const Nil& object) :
      world(fiberWorld) {
    //
  }

  WeakPointer(const WeakPointer<Any>& o) :
      object(o.object),
      world(fiberClone ? fiberWorld : o.world) {
    //
  }

  template<class U>
  WeakPointer(const WeakPointer<U>& o) :
      object(o.object),
      world(fiberWorld) {
    //
  }

  template<class U>
  WeakPointer(const SharedPointer<U>& o) :
      object(o.object),
      world(fiberWorld) {
    //
  }

  template<class U>
  WeakPointer(const Optional<SharedPointer<U>>& o) :
      WeakPointer(o.query() ? o.get() : nullptr) {
    //
  }

  WeakPointer<Any>& operator=(const WeakPointer<Any>& o) {
    assert(world.lock()->hasLaunchAncestor(o.world));
    object = o.object;
    return *this;
  }

protected:
  /**
   * The object.
   */
  std::weak_ptr<Any> object;

  /**
   * The world to which the object should belong (although it may belong to
   * a clone ancestor of this world).
   */
  std::weak_ptr<World> world;
};
}
