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
#include "libbirch/Nil.hpp"

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
  SharedCOW(T* object, Memo* memo = currentContext.get()) :
      super_type(object, memo) {
    //
  }

  /**
   * Constructor.
   */
  SharedCOW(const SharedPtr<T>& object, Memo* memo = currentContext.get()) :
      super_type(object, memo) {
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
    return static_cast<T*>(root_type::get());
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

  /**
   * Deep clone.
   */
  SharedCOW<T> clone() {
    return root_type::clone().template static_pointer_cast<T>();
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

  SharedCOW(Any* object, Memo* memo = currentContext.get()) :
      object(object),
      memo(memo) {
    //
  }

  SharedCOW(const SharedPtr<Any>& object, Memo* memo = currentContext.get()) :
      object(object),
      memo(memo) {
    //
  }

  SharedCOW(const WeakCOW<Any>& o);

  SharedCOW(const SharedCOW<Any>& o) :
      object(o.object),
      memo(o.memo) {
    if (cloneUnderway && object) {
      if (!currentContext->hasAncestor(memo.get())) {
        object = memo->get(object.get());
      }
      memo = currentContext.get();
      auto parent = memo->getParent();
      if (parent) {
        object = parent->deep(object.get());
      }
      #if !USE_LAZY_DEEP_CLONE
      get();
      #endif
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
    #if USE_LAZY_DEEP_CLONE
    object = memo->get(object.get())->getForward();
    assert(!object->isFrozen());
    #endif
    return object.get();
  }

  Any* get() const {
    /* even in a const context, do want to update the pointer through lazy
     * deep clone mechanisms */
    return const_cast<SharedCOW<Any>*>(this)->get();
  }

  Any* pull() {
    #if USE_LAZY_DEEP_CLONE
    object = memo->pull(object.get());
    if (object->getContext() == memo.get()) {
      object = object->pullForward();
    }
    #endif
    return object.get();
  }

  Any* pull() const {
    /* even in a const context, do want to update the pointer through lazy
     * deep clone mechanisms */
    return const_cast<SharedCOW<Any>*>(this)->pull();
  }

  SharedCOW<Any> clone() {
    freeze();
    SharedCOW<Any> result(object, memo->fork());
    #if !USE_LAZY_DEEP_CLONE
    result.get();
    #endif
    return result;
  }

  void freeze() {
    if (object) {
      object = memo->pull(object.get());
      object->freeze();
      memo->freeze();
    }
  }

  Memo* getContext() const {
    return memo.get();
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
    return SharedCOW<U>(dynamic_cast<U*>(object.get()), memo.get());
  }

  /**
   * Static cast. Undefined if unsuccessful.
   */
  template<class U>
  SharedCOW<U> static_pointer_cast() const {
    return SharedCOW<U>(static_cast<U*>(object.get()), memo.get());
  }

protected:
  /**
   * The object.
   */
  SharedPtr<Any> object;

  /**
   * The memo.
   */
  ContextPtr memo;
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
    memo(o.memo) {
  //
}
