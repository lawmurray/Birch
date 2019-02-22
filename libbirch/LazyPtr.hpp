/**
 * @file
 */
#pragma once
#if ENABLE_LAZY_DEEP_CLONE

#include "libbirch/clone.hpp"
#include "libbirch/LazyAny.hpp"
#include "libbirch/LazyMemo.hpp"
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
  LazyPtr(T* object, LazyMemo* from = currentContext, LazyMemo* to = currentContext) :
      object(object),
      from(from),
      to(to) {
    //
  }

  /**
   * Constructor.
   */
  LazyPtr(const P& object, LazyMemo* from = currentContext, LazyMemo* to =
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
      if (o.object && !currentContext->hasAncestor(o.to.get())) {
        *this = o.clone();
      } else {
        object = o.object;
        from = o.from;
        to = currentContext;
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
      object = static_cast<T*>(to->get(object.get(), from.get())->getForward());
      from = to.get();
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
      object = static_cast<T*>(to->pull(object.get(), from.get())->pullForward());
      if (object->getContext() == to.get()) {
        from = to.get();
      } else {
        /* copy has been omitted for this access as it is read only, but on
         * the next access we will need to check whether a copy has happened
         * elsewhere in the meantime */
        from = to->getParent();
      }
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
  LazyPtr<P> clone() const {
    freeze();
    return LazyPtr<P>(object, from.get(), to->fork());
  }

  /**
   * Freeze.
   */
  void freeze() const {
    if (object) {
      pull();
      object->freeze();
      to->freeze();
    }
  }

  /**
   * Get the context of the object.
   */
  LazyMemo* getContext() const {
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
   * First label in list. This is potentially different from LazyAny::context
   * as read-only accesses via pull() may partially propagate objects through
   * the memo list.
   */
  InitPtr<LazyMemo> from;

  /**
   * Last label in list.
   */
  ContextPtr to;
};
}

#endif
