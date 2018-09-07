/**
 * @file
 */
#pragma once

#include "libbirch/global.hpp"
#include "libbirch/SharedPtr.hpp"
#include "libbirch/World.hpp"
#include "libbirch/Any.hpp"
#include "libbirch/Nil.hpp"

namespace bi {
/**
 * Shared pointer with copy-on-write semantics.
 *
 * @ingroup libbirch
 *
 * @tparam T Type.
 */
template<class T>
class SharedCOW: public SharedCOW<typename super_type<T>::type> {
  template<class U> friend class WeakCOW;
public:
  using value_type = T;
  using this_type = SharedCOW<T>;
  using super_type = SharedCOW<typename super_type<T>::type>;
  using root_type = typename super_type::root_type;

  /**
   * Constructor.
   */
  SharedCOW(T* object = nullptr) :
      super_type(object) {
    //
  }

  /**
   * Constructor.
   */
  SharedCOW(const Nil& object) :
      super_type(object) {
    //
  }

  /**
   * Constructor.
   */
  SharedCOW(const SharedPtr<T>& object) :
      super_type(object) {
    //
  }

  /**
   * Constructor.
   */
  SharedCOW(const WeakPtr<T>& object) :
      super_type(object) {
    //
  }

  /**
   * Constructor.
   */
  SharedCOW(const WeakCOW<T>& o);

  /**
   * Constructor.
   */
  SharedCOW(T* object, World* world, World* current) :
      super_type(object, world, current) {
    //
  }

  /**
   * Copy constructor.
   */
  SharedCOW(const SharedCOW<T>& o) = default;

  /**
   * Move constructor.
   */
  SharedCOW(SharedCOW<T>&& o) = default;

  /**
   * Copy assignment.
   */
  SharedCOW<T>& operator=(const SharedCOW<T>& o) = default;

  /**
   * Move assignment.
   */
  SharedCOW<T>& operator=(SharedCOW<T>&& o) = default;

  /**
   * Generic copy assignment.
   */
  template<class U>
  SharedCOW<T>& operator=(const SharedCOW<U>& o) {
    root_type::operator=(o);
    return *this;
  }

  /**
   * Generic move assignment.
   */
  template<class U>
  SharedCOW<T>& operator=(SharedCOW<U>&& o) {
    root_type::operator=(o);
    return *this;
  }

  /**
   * Value assignment.
   */
  template<class U,
      typename = std::enable_if_t<bi::has_assignment<T,U>::value>>
  SharedCOW<T>& operator=(const U& o) {
    *get() = o;
    return *this;
  }

  /**
   * Value conversion.
   */
  template<class U,
      typename = std::enable_if_t<bi::has_conversion<T,U>::value>>
  operator U() const {
    return static_cast<U>(*get());
  }

  /**
   * Get the raw pointer.
   */
  T* get() const {
    return static_cast<T*>(root_type::get());
  }

  /**
   * Get the raw pointer while mapping, but not copying, into the current
   * world. The caller assumes responsibility for the validity of this; it is
   * used as an optimization.
   */
  const T* getNoCopy() const {
    return static_cast<const T*>(root_type::getNoCopy());
  }

  /**
   * Pull through generations.
   */
  T* pull() const {
    return static_cast<T*>(root_type::pull());
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
   * Call operator.
   */
  template<class ... Args>
  auto operator()(Args ... args) const {
    return (*get())(args...);
  }
};

template<>
class SharedCOW<Any> {
  template<class U> friend class WeakCOW;
public:
  using value_type = Any;
  using this_type = SharedCOW<value_type>;
  using root_type = this_type;

  SharedCOW(Any* object = nullptr) :
      object(object),
      world(fiberWorld),
      current(fiberWorld) {
    //
  }

  SharedCOW(const Nil& object) :
      object(nullptr),
      world(fiberWorld),
      current(fiberWorld) {
    //
  }

  SharedCOW(const SharedPtr<Any>& object) :
      object(object),
      world(fiberWorld),
      current(fiberWorld) {
    //
  }

  SharedCOW(const WeakPtr<Any>& object) :
      object(object),
      world(fiberWorld),
      current(fiberWorld) {
    //
  }

  SharedCOW(Any* object, World* world, World* current) :
      object(object),
      world(world),
      current(current) {
    //
  }

  SharedCOW(const SharedCOW<Any>& o) :
      object(o.object),
      world(fiberClone ? fiberWorld : o.world),
      current(o.current) {
    //
  }

  SharedCOW(const WeakCOW<Any>& o);

  SharedCOW(SharedCOW<Any> && o) = default;

  SharedCOW<Any>& operator=(const SharedCOW<Any>& o) {
    bi_assert_msg(world->hasLaunchAncestor(o.world),
        "when a fiber yields an object, that object cannot be kept by the caller");
    auto old = std::move(object);
    // ^ ensures next assignment doesn't destroy o

    object = o.pull();
    current = o.current;
    return *this;
  }

  SharedCOW<Any>& operator=(SharedCOW<Any>&& o) {
    bi_assert_msg(world->hasLaunchAncestor(o.world),
        "when a fiber yields an object, that object cannot be kept by the caller");
    auto old = std::move(object);
    // ^ ensures next assignment doesn't destroy o

    object = o.pull();
    current = o.current;
    return *this;
  }

  /**
   * Is the pointer not null?
   */
  bool query() const {
    return static_cast<bool>(object);
  }

  Any* get() const {
    /* despite the pointer being accessed in a const context, we do want to
     * update it through the copy-on-write mechanism for performance
     * reasons */
    if (object) {
      auto self = const_cast<SharedCOW<Any>*>(this);
      self->object = self->world->get(object.get(), current);
      self->current = self->world;
    }
    return object.get();
  }

  const Any* getNoCopy() const {
    /* despite the pointer being accessed in a const context, we do want to
     * update it through the copy-on-write mechanism for performance
     * reasons */
    if (object) {
      auto self = const_cast<SharedCOW<Any>*>(this);
      self->object = self->world->getNoCopy(object.get(), current);
      self->current = self->world;
    }
    return object.get();
  }

  Any* pull() const {
    /* despite the pointer being accessed in a const context, we do want to
     * update it through the copy-on-write mechanism for performance
     * reasons */
    if (object) {
      auto self = const_cast<SharedCOW<Any>*>(this);
      self->object = self->world->getNoCopy(object.get(), current);
      self->current = self->world;
    }
    return object.get();
  }

  World* getWorld() const {
    return world;
  }

  Any& operator*() const {
    return *get();
  }

  Any* operator->() const {
    return get();
  }

  bool operator==(const SharedCOW<Any>& o) const {
    return get() == o.get();
  }

  bool operator!=(const SharedCOW<Any>& o) const {
    return get() != o.get();
  }

  /**
   * Dynamic cast. Returns `nullptr` if unsuccessful.
   */
  template<class U>
  SharedCOW<U> dynamic_pointer_cast() const {
    return SharedCOW<U>(dynamic_cast<U*>(get()), world, current);
  }

  /**
   * Static cast. Undefined if unsuccessful.
   */
  template<class U>
  SharedCOW<U> static_pointer_cast() const {
    return SharedCOW<U>(static_cast<U*>(get()), world, current);
  }

protected:
  /**
   * The object.
   */
  SharedPtr<Any> object;

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

#include "libbirch/WeakCOW.hpp"

template<class T>
bi::SharedCOW<T>::SharedCOW(const WeakCOW<T>& o) :
    super_type(o) {
  //
}

inline bi::SharedCOW<bi::Any>::SharedCOW(const WeakCOW<Any>& o) :
    object(o.object),
    world(o.world),
    current(o.current) {
  //
}
