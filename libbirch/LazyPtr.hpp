/**
 * @file
 */
#pragma once

#include "libbirch/config.hpp"
#include "libbirch/memory.hpp"
#include "libbirch/Any.hpp"
#include "libbirch/Nil.hpp"
#include "libbirch/InitPtr.hpp"
#include "libbirch/ContextPtr.hpp"

#include <tuple>

namespace bi {
/**
 * Wraps another pointer type to apply lazy deep clone semantics.
 *
 * @ingroup libbirch
 *
 * @tparam P Pointer type.
 */
template<class P>
class LazyPtr {
  template<class U> friend class LazyPtr;
public:
  using T = typename P::value_type;
  template<class U> using cast_type = LazyPtr<typename P::template cast_type<U>>;

  /**
   * Constructor.
   */
  LazyPtr(const Nil& = nil) {
    //
  }

  /**
   * Constructor.
   */
  LazyPtr(T* object, Memo* from = currentContext, Memo* to = currentContext) :
      object(object),
      from(from),
      to(to) {
    //
  }

  /**
   * Constructor.
   */
  LazyPtr(const P& object, Memo* from = currentContext, Memo* to =
      currentContext) :
      object(object),
      from(from),
      to(to) {
    //
  }

  /**
   * Copy constructor.
   */
  LazyPtr(const LazyPtr<P>& o) {
    if (cloneUnderway) {
      to = currentContext;
      if (o.object && !currentContext->hasAncestor(o.to.get())) {
        Any* tmp;
        std::tie(tmp, from) = currentContext->getNoForward(o.object.get(),
            o.from.get());
        object = static_cast<T*>(tmp);
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

  /**
   * Generic copy constructor.
   */
  template<class Q>
  LazyPtr(const LazyPtr<Q>& o) :
      object(o.object),
      from(o.from),
      to(o.to) {
    //
  }

  LazyPtr(LazyPtr<P> && o) = default;
  LazyPtr<P>& operator=(const LazyPtr<P>& o) = default;
  LazyPtr<P>& operator=(LazyPtr<P> && o) = default;

  /**
   * Generic copy assignment.
   */
  template<class Q>
  LazyPtr<P>& operator=(const LazyPtr<Q>& o) {
    object = o.object;
    from = o.from;
    to = o.to;
    return *this;
  }

  /**
   * Generic move assignment.
   */
  template<class Q>
  LazyPtr<P>& operator=(LazyPtr<Q> && o) {
    object = o.object;
    from = o.from;
    to = o.to;
    return *this;
  }

  /**
   * Value assignment.
   */
  template<class U,
      typename = std::enable_if_t<bi::has_assignment<T,U>::value>>
  LazyPtr<P>& operator=(const U& o) {
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
   * Is the pointer not null?
   */
  bool query() const {
    return static_cast<bool>(object);
  }

  /**
   * Get the raw pointer, lazy cloning if necessary.
   */
  T* get() {
    if (object) {
      Any* tmp;
      std::tie(tmp, from) = to->get(object.get(), from.get());
      object = static_cast<T*>(tmp);
      assert(!object->isFrozen());
    }
    return object.get();
  }

  /**
   * Get the raw pointer, lazy cloning if necessary.
   */
  T* get() const {
    /* even in a const context, do want to update the pointer through lazy
     * deep clone mechanisms */
    return const_cast<LazyPtr<P>*>(this)->get();
  }

  /**
   * Map the raw pointer, without lazy cloning.
   */
  const T* pull() {
    if (object) {
      Any* tmp;
      std::tie(tmp, from) = to->pull(object.get(), from.get());
      object = static_cast<T*>(tmp);
    }
    return object.get();
  }

  /**
   * Map the raw pointer, without lazy cloning.
   */
  const T* pull() const {
    /* even in a const context, do want to update the pointer through lazy
     * deep clone mechanisms */
    return const_cast<LazyPtr<P>*>(this)->pull();
  }

  /**
   * Deep clone.
   */
  LazyPtr<P> clone() {
    freeze();
    return LazyPtr<P>(object, from.get(), to->fork());
  }

  /**
   * Deep clone.
   */
  LazyPtr<P> clone() const {
    return const_cast<LazyPtr<P>*>(this)->clone();
  }

  /**
   * Freeze.
   */
  void freeze() {
    if (object) {
      Any* tmp;
      std::tie(tmp, from) = to->pull(object.get(), from.get());
      object = static_cast<T*>(tmp);
      object->freeze();
      to->freeze();
    }
  }

  /**
   * Get the context of the object.
   */
  Memo* getContext() const {
    return to.get();
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
   * Equal comparison.
   */
  template<class U>
  bool operator==(const LazyPtr<U>& o) const {
    return get() == o.get();
  }

  /**
   * Not equal comparison.
   */
  template<class U>
  bool operator!=(const LazyPtr<U>& o) const {
    return get() != o.get();
  }

  /**
   * Dynamic cast. Returns `nullptr` if unsuccessful.
   */
  template<class U>
  auto dynamic_pointer_cast() const {
    return cast_type<U>(dynamic_cast<U*>(object.get()), from.get(), to.get());
  }

  /**
   * Static cast. Undefined if unsuccessful.
   */
  template<class U>
  auto static_pointer_cast() const {
    return cast_type<U>(static_cast<U*>(object.get()), from.get(), to.get());
  }

protected:
  /**
   * Object.
   */
  P object;

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
