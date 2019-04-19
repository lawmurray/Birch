/**
 * @file
 */
#pragma once
#if ENABLE_LAZY_DEEP_CLONE

#include "libbirch/clone.hpp"
#include "libbirch/LazyAny.hpp"
#include "libbirch/LazyMemo.hpp"
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
  LazyPtr(T* object, LazyMemo* to) :
      object(object),
      to(to) {
    //
  }

  /**
   * Constructor.
   */
  LazyPtr(const P& object, LazyMemo* to) :
      object(object),
      to(to) {
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
      to = currentContext;
    } else {
      object = o.object;
      to = o.to;
    }
  }

  /**
   * Generic copy constructor.
   */
  template<class Q>
  LazyPtr(const LazyPtr<Q>& o) :
      object(o.object),
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
    to = o.to;
    return *this;
  }

  /**
   * Generic move assignment.
   */
  template<class Q>
  LazyPtr<P>& operator=(LazyPtr<Q> && o) {
    object = o.object;
    to = o.to;
    return *this;
  }

  /**
   * Raw pointer assignment.
   */
  LazyPtr<P>& operator=(T* o) {
    object = o;
    to = currentContext;
    return *this;
  }

  /**
   * Nil assignment.
   */
  LazyPtr<P>& operator=(const Nil&) {
    object = nullptr;
    to = nullptr;
    return *this;
  }

  /**
   * Nullptr assignment.
   */
  LazyPtr<P>& operator=(const std::nullptr_t&) {
    object = nullptr;
    to = nullptr;
    return *this;
  }

  /**
   * Optional assignment.
   */
  template<class Q>
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
    if (object && object->isFrozen()) {
      LazyAny* raw = object.get();
      if (to) {
        raw = to->get(raw);
        if (raw->isFrozen()) {
          raw = raw->getForward();
        }
      } else {
        raw = raw->getForward();
      }
      assert(!raw->isFrozen());
      object = static_cast<T*>(raw);
    }
    return object.get();
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
    if (object && object->isFrozen()) {
      LazyAny* raw = object.get();
      if (to) {
        raw = to->pull(raw);
        if (raw->getContext() == to.get() && raw->isFrozen()) {
          raw = raw->pullForward();
        }
      } else {
        raw = raw->pullForward();
      }
      object = static_cast<T*>(raw);
    }
    return object.get();
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
    freeze();
    LazyMemo* memo;
    if (to) {
      memo = to->fork();
    } else {
      memo = LazyMemo::create_();
    }
    return LazyPtr<P>(object, memo);
  }

  /**
   * Freeze.
   */
  void freeze() const {
    if (object) {
      pull();
      object->freeze();
      if (to) {
        to->freeze();
      }
    }
  }

  /**
   * Finish.
   */
  void finish() {
    if (object) {
      object = static_cast<T*>(to->get(object.get()));
      object->finish();
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
