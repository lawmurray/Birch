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
  LazyPtr(const LazyPtr<P>& o) :
      object(nullptr),
      to(cloneUnderway ? currentContext : o.to) {
    if (o.object) {
      if (cloneUnderway) {
        if (o.isCross()) {
          o.finish();
        }
        object = o.object;
      } else {
        object = o.object->isSingular() ? o.get() : o.object;
      }
    }
  }

  /**
   * Generic copy constructor.
   */
  template<class Q, typename = std::enable_if_t<std::is_base_of<T,
      typename Q::value_type>::value>>
  LazyPtr(const LazyPtr<Q>& o) :
      object((o.object && o.object->isSingular()) ? o.get() : o.object),
      to(o.object ? o.to : nullptr) {
    //
  }

  /**
   * Move constructor.
   */
  LazyPtr(LazyPtr<P> && o) = default;

  /**
   * Generic move constructor.
   */
  template<class Q>
  LazyPtr(LazyPtr<Q> && o) :
      object(std::move(o.object)),
      to(std::move(o.to)) {
    //
  }

  /**
   * Copy assignment.
   */
  LazyPtr<P>& operator=(const LazyPtr<P>& o) {
    /* it's possible that object = o.object actually destroys the referent
     * of o, so do the to = o.to first */
    to = o.to;
    object = (o.object && o.object->isSingular()) ? o.get() : o.object;
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
    to = o.to;
    object = (o.object && o.object->isSingular()) ? o.get() : o.object;
    return *this;
  }

  /**
   * Move assignment.
   */
  LazyPtr<P>& operator=(LazyPtr<P> && o) {
    /* it's possible that object = o.object actually destroys the referent
     * of o, so do the to = o.to first */
    to = std::move(o.to);
    object = std::move(o.object);
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
    to = std::move(o.to);
    object = std::move(o.object);
    return *this;
  }

  /**
   * Raw pointer assignment.
   */
  LazyPtr<P>& operator=(T* o) {
    assert(!o || !o->isSingular());
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
    auto raw = object.get();
    if (raw && raw->isFrozen()) {
      raw = static_cast<T*>(to->pull(raw));
      object = raw;
    }
    return raw;
  }

  /**
   * Get the raw pointer for read-only use, without cloning.
   */
  const T* pull() const {
    return const_cast<LazyPtr<P>*>(this)->pull();
  }

  /**
   * Get the raw pointer for possibly read-only use. Calls get() or pull()
   * according to whether ENABLE_READ_ONLY_OPTIMIZATION is true or false,
   * respectively.
   */
  const T* readOnly() {
    #if ENABLE_READ_ONLY_OPTIMIZATION
    return pull();
    #else
    return get();
    #endif
  }

  /**
   * Get the raw pointer for possibly read-only use. Calls get() or pull()
   * according to whether ENABLE_READ_ONLY_OPTIMIZATION is true or false,
   * respectively.
   */
  const T* readOnly() const {
    return const_cast<LazyPtr<P>*>(this)->readOnly();
  }

  /**
   * Deep clone.
   */
  LazyPtr<P> clone() const {
    assert(object);
    pull();
    freeze();
    return LazyPtr<P>(object, to->fork());
  }

  /**
   * Freeze.
   */
  void freeze() {
    if (object) {
      bool top = !freezeUnderway;  // is this the top call of the recursion?
      if (top) {
        freezeUnderway = true;
        freezeLock.enter();
      }
      object->freeze();
      to->freeze();
      if (top) {
        freezeLock.exit();
        freezeUnderway = false;
      }
    }
  }

  /**
   * Freeze.
   */
  void freeze() const {
    return const_cast<LazyPtr<P>*>(this)->freeze();
  }

  /**
   * Finish.
   */
  void finish() {
    if (object) {
      bool top = !finishUnderway;  // is this the top call of the recursion?
      if (top) {
        finishUnderway = true;
        finishLock.enter();
      }
      get();
      object->finish();
      if (top) {
        finishLock.exit();
        finishUnderway = false;
      }
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
