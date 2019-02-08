/**
 * @file
 */
#pragma once

#include "libbirch/config.hpp"
#include "libbirch/class.hpp"
#include "libbirch/memory.hpp"
#include "libbirch/SharedPtr.hpp"
#include "libbirch/ContextPtr.hpp"
#include "libbirch/Any.hpp"

#include <tuple>

namespace bi {
template<class U> class WeakCOW;

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
  using super_type = SharedCOW<typename super_type<T>::type>;
  using root_type = typename super_type::root_type;

  /**
   * Constructor.
   */
  SharedCOW(const Nil& = nil) {
    //
  }

  /**
   * Constructor.
   */
  SharedCOW(T* object, Memo* from = currentContext, Memo* to = currentContext) :
      super_type(object, from, to) {
    //
  }

  /**
   * Constructor.
   */
  SharedCOW(const SharedPtr<T>& object, Memo* from = currentContext,
      Memo* to = currentContext) :
      super_type(object, from, to) {
    //
  }

  /**
   * Constructor.
   */
  SharedCOW(const WeakCOW<T>& o);

  SharedCOW(const SharedCOW<T>& o) = default;
  SharedCOW(SharedCOW<T> && o) = default;
  SharedCOW<T>& operator=(const SharedCOW<T>& o) = default;
  SharedCOW<T>& operator=(SharedCOW<T> && o) = default;

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
  SharedCOW<T>& operator=(SharedCOW<U> && o) {
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
   * Get the raw pointer, lazy cloning if necessary.
   */
  T* get() {
    return static_cast<T*>(root_type::get());
  }

  /**
   * Get the raw pointer, lazy cloning if necessary.
   */
  T* get() const {
    return const_cast<SharedCOW<T>*>(this)->get();
  }

  /**
   * Map the raw pointer, without lazy cloning.
   */
  const T* pull() {
    return static_cast<const T*>(root_type::pull());
  }

  /**
   * Map the raw pointer, without lazy cloning.
   */
  const T* pull() const {
    return const_cast<SharedCOW<T>*>(this)->pull();
  }

  /**
   * Deep clone.
   */
  SharedCOW<T> clone() {
    return root_type::clone().template static_pointer_cast<T>();
  }

  /**
   * Deep clone.
   */
  SharedCOW<T> clone() const {
    return const_cast<SharedCOW<T>*>(this)->clone();
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

template<>
class SharedCOW<Any> {
  template<class U> friend class WeakCOW;
public:
  using value_type = Any;
  using root_type = SharedCOW<value_type>;

  SharedCOW(const Nil& = nil) {
    //
  }

  SharedCOW(Any* object, Memo* from = currentContext, Memo* to =
      currentContext) :
      object(object),
      from(from),
      to(to) {
    //
  }

  SharedCOW(const SharedPtr<Any>& object, Memo* from = currentContext,
      Memo* to = currentContext) :
      object(object),
      from(from),
      to(to) {
    //
  }

  SharedCOW(const WeakCOW<Any>& o);

  SharedCOW(const SharedCOW<Any>& o) {
    if (cloneUnderway) {
      to = currentContext;
      if (o.object && !currentContext->hasAncestor(o.to.get())) {
        std::tie(object, from) = currentContext->getNoForward(o.object.get(), o.from.get());
        freeze();
      } else {
        object = o.object;
        from = o.from;
      }
    } else {
      object = o.object;
      from = o.from;
      to = o.to;
    }
  }

  SharedCOW(SharedCOW<Any> && o) = default;
  SharedCOW<Any>& operator=(const SharedCOW<Any>& o) = default;
  SharedCOW<Any>& operator=(SharedCOW<Any> && o) = default;

  /**
   * Is the pointer not null?
   */
  bool query() const {
    return static_cast<bool>(object);
  }

  Any* get() {
    if (object) {
      std::tie(object, from) = to->get(object.get(), from.get());
      assert(!object->isFrozen());
    }
    return object.get();
  }

  Any* get() const {
    /* even in a const context, do want to update the pointer through lazy
     * deep clone mechanisms */
    return const_cast<SharedCOW<Any>*>(this)->get();
  }

  const Any* pull() {
    if (object) {
      std::tie(object, from) = to->pull(object.get(), from.get());
    }
    return object.get();
  }

  const Any* pull() const {
    /* even in a const context, do want to update the pointer through lazy
     * deep clone mechanisms */
    return const_cast<SharedCOW<Any>*>(this)->pull();
  }

  SharedCOW<Any> clone() {
    freeze();
    return SharedCOW<Any>(object, from.get(), to->fork());
  }

  SharedCOW<Any> clone() const {
    return const_cast<SharedCOW<Any>*>(this)->clone();
  }

  void freeze() {
    if (object) {
      std::tie(object, from) = to->pull(object.get(), from.get());
      object->freeze();
      to->freeze();
    }
  }

  Memo* getContext() const {
    return to.get();
  }

  Any& operator*() const {
    return *get();
  }

  Any* operator->() const {
    return get();
  }

  template<class U>
  bool operator==(const SharedCOW<U>& o) const {
    return get() == o.get();
  }

  template<class U>
  bool operator!=(const SharedCOW<U>& o) const {
    return get() != o.get();
  }

  /**
   * Dynamic cast. Returns `nullptr` if unsuccessful.
   */
  template<class U>
  SharedCOW<U> dynamic_pointer_cast() const {
    return SharedCOW<U>(dynamic_cast<U*>(object.get()), from.get(), to.get());
  }

  /**
   * Static cast. Undefined if unsuccessful.
   */
  template<class U>
  SharedCOW<U> static_pointer_cast() const {
    return SharedCOW<U>(static_cast<U*>(object.get()), from.get(), to.get());
  }

protected:
  /**
   * Object.
   */
  SharedPtr<Any> object;

  /**
   * First label in list.
   */
  InitPtr<Memo> from;

  /**
   * Last label in list.
   */
  ContextPtr to;
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
    from(o.from),
    to(o.to) {
  //
}
