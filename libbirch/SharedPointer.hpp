/**
 * @file
 */
#pragma once

#include "libbirch/global.hpp"
#include "libbirch/World.hpp"
#include "libbirch/Any.hpp"
#include "libbirch/Nil.hpp"

namespace bi {
template<class T> class SharedPointer;
template<class T> class WeakPointer;

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
   * Constructor.
   */
  SharedPointer(const std::nullptr_t& object = nullptr, World* world =
      fiberWorld) :
      super_type(object, world) {
    //
  }

  /**
   * Constructor.
   */
  SharedPointer(const Nil& object, World* world = fiberWorld) :
      super_type(object, world) {
    //
  }

  /**
   * Constructor.
   */
  SharedPointer(const std::shared_ptr<T>& object) :
      super_type(object) {
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
  std::shared_ptr<T> pull() const {
    return std::static_pointer_cast<T>(root_type::pull());
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
class SharedPointer<Any> {
  template<class U> friend class SharedPointer;
  template<class U> friend class WeakPointer;
public:
  using value_type = Any;
  using this_type = SharedPointer<value_type>;
  using root_type = this_type;

  SharedPointer(const std::nullptr_t& object = nullptr, World* world =
      fiberWorld) :
      world(world) {
    //
  }

  SharedPointer(const Nil& object, World* world = fiberWorld) :
      world(world) {
    //
  }

  SharedPointer(const std::shared_ptr<Any>& object) :
      object(object),
      world(fiberWorld) {
    //
  }

  SharedPointer(const SharedPointer<Any>& o) :
      object(o.pull()),
      world(fiberClone ? fiberWorld : o.world) {
    //
  }

  template<class U>
  SharedPointer(const SharedPointer<U>& o) :
      object(o.pull()),
      world(o.world) {
    //
  }

  template<class U>
  SharedPointer(const WeakPointer<U>& o) :
      object(o.pull()),
      world(o.world) {
    //
  }

  SharedPointer<Any>& operator=(const SharedPointer<Any>& o) {
    bi_assert_msg(world->hasLaunchAncestor(o.world), "when a fiber yields an object, that object cannot be kept by the caller");
    object = o.pull();
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
      auto self = const_cast<SharedPointer<Any>*>(this);
      self->object = self->world->get(object);
    }
    return object.get();
  }

  const Any* getNoCopy() const {
    /* despite the pointer being accessed in a const context, we do want to
     * update it through the copy-on-write mechanism for performance
     * reasons */
    if (object) {
      auto self = const_cast<SharedPointer<Any>*>(this);
      self->object = self->world->getNoCopy(object);
    }
    return object.get();
  }

  std::shared_ptr<Any> pull() const {
    /* despite the pointer being accessed in a const context, we do want to
     * update it through the copy-on-write mechanism for performance
     * reasons */
    if (object) {
      auto self = const_cast<SharedPointer<Any>*>(this);
      self->object = self->world->getNoCopy(object);
    }
    return object;
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

  bool operator==(const SharedPointer<Any>& o) const {
    return get() == o.get();
  }

  bool operator!=(const SharedPointer<Any>& o) const {
    return get() != o.get();
  }

  /**
   * Dynamic cast. Returns `nullptr` if the cast if unsuccessful.
   */
  template<class U>
  SharedPointer<U> dynamic_pointer_cast() const {
    return SharedPointer<U>(std::dynamic_pointer_cast < U > (object));
  }

  /**
   * Static cast. Undefined if unsuccessful.
   */
  template<class U>
  SharedPointer<U> static_pointer_cast() const {
    return SharedPointer<U>(std::static_pointer_cast < U > (object));
  }

protected:
  /**
   * The object.
   */
  std::shared_ptr<Any> object;

  /**
   * The world to which the object should belong (although it may belong to
   * a clone ancestor of this world).
   */
  World* world;
};
}
