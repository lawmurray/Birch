/**
 * @file
 */
#pragma once

#include "libbirch/config.hpp"
#include "libbirch/clone.hpp"
#include "libbirch/SharedCOW.hpp"
#include "libbirch/WeakPtr.hpp"
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
class WeakCOW: public WeakCOW<typename super_type<T>::type> {
  template<class U> friend class SharedCOW;
public:
  using value_type = T;
  using super_type = WeakCOW<typename super_type<T>::type>;
  using root_type = typename super_type::root_type;

  /**
   * Constructor.
   */
  WeakCOW(const Nil& = nil) :
      super_type() {
    //
  }

  /**
   * Constructor.
   */
  WeakCOW(const SharedCOW<T>& o) :
      super_type(o) {
    //
  }

  WeakCOW(const WeakCOW<T>& o) = default;
  WeakCOW(WeakCOW<T>&& o) = default;
  WeakCOW<T>& operator=(const WeakCOW<T>& o) = default;
  WeakCOW<T>& operator=(WeakCOW<T>&& o) = default;

  /**
   * Generic copy assignment.
   */
  template<class U>
  WeakCOW<T>& operator=(const WeakCOW<U>& o) {
    root_type::operator=(o);
    return *this;
  }

  /**
   * Generic move assignment.
   */
  template<class U>
  WeakCOW<T>& operator=(WeakCOW<U>&& o) {
    root_type::operator=(o);
    return *this;
  }

  /**
   * Generic copy assignment.
   */
  template<class U>
  WeakCOW<T>& operator=(const SharedCOW<U>& o) {
    root_type::operator=(o);
    return *this;
  }

  /**
   * Generic move assignment.
   */
  template<class U>
  WeakCOW<T>& operator=(SharedCOW<U>&& o) {
    root_type::operator=(o);
    return *this;
  }

  /**
   * Map the raw pointer, without lazy cloning.
   */
  T* pull() {
    return static_cast<T*>(root_type::pull());
  }

  /**
   * Map the raw pointer, without lazy cloning.
   */
  T* pull() const {
    return static_cast<T*>(root_type::pull());
  }
};

template<>
class WeakCOW<Any> {
  template<class U> friend class SharedCOW;
public:
  using value_type = Any;
  using root_type = WeakCOW<value_type>;

  WeakCOW(const Nil& = nil) {
    //
  }

  WeakCOW(const SharedCOW<Any>& o) :
      object(o.object),
      memo(o.memo) {
    //
  }

  WeakCOW(const WeakCOW<Any>& o) :
      object(o.object),
      memo(o.memo) {
    if (cloneUnderway && object) {
      clone_continue(object, memo);
    }
  }

  WeakCOW(WeakCOW<Any> && o) = default;
  WeakCOW<Any>& operator=(const WeakCOW<Any>& o) = default;
  WeakCOW<Any>& operator=(WeakCOW<Any>&& o) = default;

  Any* pull() {
    #if DEEP_CLONE_STRATEGY != DEEP_CLONE_EAGER
    if (object) {
      memo = memo->forwardPull();
      clone_pull(object, memo);
    }
    #endif
    return object.get();
  }

  Any* pull() const {
    /* even in a const context, do want to update the pointer through lazy
     * deep clone mechanisms */
   return const_cast<WeakCOW<Any>*>(this)->pull();
  }

protected:
  /**
   * The object.
   */
  WeakPtr<Any> object;

  /**
   * The memo.
   */
  SharedPtr<Memo> memo;
};
}
