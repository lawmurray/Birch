/**
 * @file
 */
#pragma once
#if !ENABLE_LAZY_DEEP_CLONE

#include "libbirch/clone.hpp"
#include "libbirch/EagerMemo.hpp"
#include "libbirch/Nil.hpp"
#include "libbirch/SwapClone.hpp"
#include "libbirch/SwapContext.hpp"

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
class EagerPtr {
  template<class U> friend class EagerPtr;
  public:
  using T = typename P::value_type;
  template<class U> using cast_type = EagerPtr<typename P::template cast_type<U>>;

  /**
   * Constructor.
   */
  EagerPtr(const Nil& = nil) {
    //
  }

  /**
   * Constructor.
   */
  EagerPtr(T* object) :
      object(object) {
    //
  }

  /**
   * Constructor.
   */
  EagerPtr(const P& object) :
      object(object) {
    //
  }

  /**
   * Copy constructor.
   */
  EagerPtr(const EagerPtr<P>& o) {
    if (cloneUnderway && o.object) {
      object = static_cast<T*>(currentContext->get(o.get()));
    } else {
      object = o.object;
    }
  }

  /**
   * Generic copy constructor.
   */
  template<class Q>
  EagerPtr(const EagerPtr<Q>& o) :
      object(o.object) {
    //
  }

  EagerPtr(EagerPtr<P> && o) = default;
  EagerPtr<P>& operator=(const EagerPtr<P>& o) = default;
  EagerPtr<P>& operator=(EagerPtr<P> && o) = default;

  /**
   * Generic copy assignment.
   */
  template<class Q>
  EagerPtr<P>& operator=(const EagerPtr<Q>& o) {
    object = o.object;
    return *this;
  }

  /**
   * Generic move assignment.
   */
  template<class Q>
  EagerPtr<P>& operator=(EagerPtr<Q> && o) {
    object = o.object;
    return *this;
  }

  /**
   * Value assignment.
   */
  template<class U,
      typename = std::enable_if_t<bi::has_assignment<T,U>::value>>
  EagerPtr<P>& operator=(const U& o) {
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
   * Get the raw pointer.
   */
  T* get() {
    return object.get();
  }

  /**
   * Get the raw pointer.
   */
  T* get() const {
    return object.get();
  }

  /**
   * Get the raw pointer as const.
   */
  const T* pull() {
    return object.get();
  }

  /**
   * Get the raw pointer as const.
   */
  const T* pull() const {
    return object.get();
  }

  /**
   * Deep clone.
   */
  EagerPtr<P> clone() const {
    if (object) {
      SharedPtr<EagerMemo> memo = EagerMemo::create();
      SwapClone swapClone(true);
      SwapContext swapContext(memo.get());
      return EagerPtr<P>(static_cast<T*>(currentContext->copy(object.get())));
    } else {
      return EagerPtr<P>();
    }
  }

  /**
   * Freeze.
   *
   * @note Should never be called, but here for interface compatibility with
   * LazyPtr.
   */
  void freeze() {
    assert(false);
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
  bool operator==(const EagerPtr<U>& o) const {
    return get() == o.get();
  }

  /**
   * Not equal comparison.
   */
  template<class U>
  bool operator!=(const EagerPtr<U>& o) const {
    return get() != o.get();
  }

  /**
   * Dynamic cast. Returns `nullptr` if unsuccessful.
   */
  template<class U>
  auto dynamic_pointer_cast() const {
    return cast_type<U>(dynamic_cast<U*>(get()));
  }

  /**
   * Static cast. Undefined if unsuccessful.
   */
  template<class U>
  auto static_pointer_cast() const {
    return cast_type<U>(static_cast<U*>(get()));
  }

protected:
  /**
   * Object.
   */
  P object;
};
}

#endif
