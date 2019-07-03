/**
 * @file
 */
#pragma once
#if ENABLE_LAZY_DEEP_CLONE

#include "libbirch/clone.hpp"
#include "libbirch/LazyAny.hpp"
#include "libbirch/LazyContext.hpp"
#include "libbirch/Nil.hpp"
#include "libbirch/ContextPtr.hpp"

namespace libbirch {
template<class T> class Optional;

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
  LazyPtr(T* object) :
      object(object),
      to(object ? currentContext : nullptr) {
    //
  }

  /**
   * Constructor.
   */
  LazyPtr(const P& object) :
      object(object),
      to(object ? currentContext : nullptr) {
    //
  }

  /**
   * Constructor.
   */
  LazyPtr(T* object, LazyContext* to) :
      object(object),
      to(object ? to : nullptr) {
    //
  }

  /**
   * Constructor.
   */
  LazyPtr(const P& object, LazyContext* to) :
      object(object),
      to(object ? to : nullptr) {
    //
  }

  /**
   * Copy constructor.
   */
  LazyPtr(const LazyPtr<P>& o) {
    if (cloneUnderway) {
      if (o.object && o.isCross()) {
        o.finish();
      }
      object = o.object;
      if (object) {
        to = currentContext;
      }
    } else if (o.object) {
      object = o.object->isUniquelyReachable() ? o.get() : o.object;
      to = o.to;
    }
  }

  /**
   * Generic copy constructor.
   */
  template<class Q, typename = std::enable_if_t<std::is_base_of<T,
      typename Q::value_type>::value>>
  LazyPtr(const LazyPtr<Q>& o) {
    if (o.object) {
      object = o.object->isUniquelyReachable() ? o.get() : o.object;
      to = o.to;
    }
  }

  /**
   * Move constructor.
   */
  LazyPtr(LazyPtr<P> && o) = default;

  /**
   * Copy assignment operator.
   */
  LazyPtr<P>& operator=(const LazyPtr<P>& o) {
    /* it's possible that object = o.object actually destroys the referent
     * of o, so do the to = o.to first */
    if (o.object) {
      to = o.to;
      object = o.object->isUniquelyReachable() ? o.get() : o.object;
    } else {
      to = nullptr;
      object = nullptr;
    }
    return *this;
  }

  /**
   * Move assignment operator.
   */
  LazyPtr<P>& operator=(LazyPtr<P> && o) {
    /* it's possible that object = o.object actually destroys the referent
     * of o, so do the to = o.to first */
    if (o.object) {
      to = o.to;
      object = std::move(o.object);
    } else {
      to = nullptr;
      object = nullptr;
    }
    return *this;
  }

  /**
   * Generic copy assignment.
   */
  template<class Q, typename = std::enable_if_t<std::is_base_of<T,
      typename Q::value_type>::value>>
  LazyPtr<P>& operator=(const LazyPtr<Q>& o) {
    /* it's possible that object = o.object actually destroys the referent
     * of o, so do the to = o.to first */
    if (o.object) {
      to = o.to;
      object = o.object->isUniquelyReachable() ? o.get() : o.object;
    } else {
      to = nullptr;
      object = nullptr;
    }
    return *this;
  }

  /**
   * Generic move assignment.
   */
  template<class Q, typename = std::enable_if_t<std::is_base_of<T,
      typename Q::value_type>::value>>
  LazyPtr<P>& operator=(LazyPtr<Q> && o) {
    /* it's possible that object = o.object actually destroys the referent
     * of o, so do the to = o.to first */
    if (o.object) {
      to = o.to;
      object = std::move(o.object);
    } else {
      to = nullptr;
      object = nullptr;
    }
    return *this;
  }

  /**
   * Raw pointer assignment.
   */
  LazyPtr<P>& operator=(T* o) {
    to = o ? currentContext : nullptr;
    object = o;
    return *this;
  }

  /**
   * Nil assignment.
   */
  LazyPtr<P>& operator=(const Nil&) {
    to = nullptr;
    object = nullptr;
    return *this;
  }

  /**
   * Nullptr assignment.
   */
  LazyPtr<P>& operator=(const std::nullptr_t&) {
    to = nullptr;
    object = nullptr;
    return *this;
  }

  /**
   * Optional assignment.
   */
  LazyPtr<P>& operator=(const Optional<LazyPtr<P>>& o) {
    if (o.query()) {
      *this = o.get();
    } else {
      *this = nullptr;
    }
    return *this;
  }

  /**
   * Generic optional assignment.
   */
  template<class Q, typename = std::enable_if_t<std::is_base_of<T,
      typename Q::value_type>::value>>
  LazyPtr<P>& operator=(const Optional<LazyPtr<Q>>& o) {
    if (o.query()) {
      *this = o.get();
    } else {
      *this = nullptr;
    }
    return *this;
  }

  /**
   * Value assignment.
   */
  template<class U>
  LazyPtr<P>& operator=(const U& o) {
    *get() = o;
    return *this;
  }

  /**
   * Value conversion.
   */
  template<class U, typename = std::enable_if_t<std::is_convertible<T,U>::value>>
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
   * Get the raw pointer, with lazy cloning.
   */
  T* get() {
    auto raw = object.get();
    if (raw && raw->isFrozen()) {
      raw = static_cast<T*>(to->get(raw));
      object = raw;
    }
    return raw;
  }

  /**
   * Get the raw pointer, with lazy cloning.
   */
  T* get() const {
    return const_cast<LazyPtr<P>*>(this)->get();
  }

  /**
   * Get the raw pointer for read-only use, without cloning.
   */
  const T* pull() {
    #if ENABLE_READ_ONLY_OPTIMIZATION
    auto raw = object.get();
    if (raw && raw->isFrozen()) {
      raw = static_cast<T*>(to->pull(raw));
      object = raw;
    }
    return raw;
    #else
    return get();
    #endif
  }

  /**
   * Get the raw pointer for read-only use, without cloning.
   */
  const T* pull() const {
    return const_cast<LazyPtr<P>*>(this)->pull();
  }

  /**
   * Deep clone.
   */
  LazyPtr<P> clone() const {
    LazyContext* context = nullptr;
    if (object) {
      freeze();
      context = to->fork();
    }
    return LazyPtr<P>(object, context);
  }

  /**
   * Freeze.
   */
  void freeze() const {
    if (object) {
      auto raw = object.get();
      if (raw->isFrozen()) {
        raw = static_cast<T*>(to->pull(raw));
      }
      raw->freeze();
      to->freeze();
    }
  }

  /**
   * Finish.
   */
  void finish() {
    auto raw = object.get();
    if (raw) {
      if (raw->isFrozen()) {
        raw = static_cast<T*>(to->finish(raw));
        object = raw;
      }
      raw->finish();
    }
  }

  /**
   * Finish.
   */
  void finish() const {
    return const_cast<LazyPtr<P>*>(this)->finish();
  }

  /**
   * Does this pointer result from a cross copy?
   */
  bool isCross() const {
    return to.get() != to.getContext();
  }

  /**
   * Get the context of the object.
   */
  LazyContext* getContext() const {
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
    return cast_type<U>(dynamic_cast<U*>(object.get()), to.get());
  }

  /**
   * Static cast. Undefined if unsuccessful.
   */
  template<class U>
  auto static_pointer_cast() const {
    return cast_type<U>(static_cast<U*>(object.get()), to.get());
  }

protected:
  /**
   * Object.
   */
  P object;

  /**
   * Context to which to map the object.
   */
  ContextPtr to;
};
}

#endif
