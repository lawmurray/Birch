/**
 * @file
 */
#pragma once

#include "libbirch/config.hpp"
#include "libbirch/SharedCOW.hpp"
#include "libbirch/WeakPtr.hpp"
#include "libbirch/ContextPtr.hpp"
#include "libbirch/Optional.hpp"

namespace bi {
/**
 * Weak pointer with copy-on-write semantics.
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
  WeakCOW(const Nil& = nil) {
    //
  }

  /**
   * Constructor.
   */
  WeakCOW(T* object, Memo* memo = top_context()) :
      super_type(object, memo) {
    //
  }

  /**
   * Constructor.
   */
  WeakCOW(const WeakPtr<T>& object, Memo* memo = top_context()) :
      super_type(object, memo) {
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
  WeakCOW(WeakCOW<T> && o) = default;
  WeakCOW<T>& operator=(const WeakCOW<T>& o) = default;
  WeakCOW<T>& operator=(WeakCOW<T> && o) = default;

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
  WeakCOW<T>& operator=(WeakCOW<U> && o) {
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
  WeakCOW<T>& operator=(SharedCOW<U> && o) {
    root_type::operator=(o);
    return *this;
  }

  /**
   * Value assignment.
   */
  template<class U,
      typename = std::enable_if_t<bi::has_assignment<T,U>::value>>
  WeakCOW<T>& operator=(const U& o) {
    *get() = o;
    return *this;
  }

  /**
   * Value conversion.
   */
  template<class U,
      typename = std::enable_if_t<bi::has_conversion<T,U>::value>>
  operator U() const {
    /* the code generator does not yet handle the push and pop of context in
     * this case */
    push_context(this->memo.get());
    auto result = static_cast<U>(*get());
    pop_context();
    return result;
  }

  /**
   * Get the raw pointer, lazy cloning if necessary.
   *
   * @warning This is inherently unsafe. The safe way to access the pointer
   * is to promote it to SharedCOW first. This is provided for convenience in
   * situations where a separate shared pointer is known to exist.
   */
  T* get() {
    return static_cast<T*>(root_type::get());
  }

  /**
   * Get the raw pointer, lazy cloning if necessary.
   *
   * @warning This is inherently unsafe. The safe way to access the pointer
   * is to promote it to SharedCOW first. This is provided for convenience in
   * situations where a separate shared pointer is known to exist.
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
class WeakCOW<Any> {
  template<class U> friend class SharedCOW;
public:
  using value_type = Any;
  using root_type = WeakCOW<value_type>;

  WeakCOW(const Nil& = nil) {
    //
  }

  WeakCOW(Any* object, Memo* memo = top_context()) :
      object(object),
      memo(memo) {
    //
  }

  WeakCOW(const WeakPtr<Any>& object, Memo* memo = top_context()) :
      object(object),
      memo(memo) {
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
      if (!top_context()->hasAncestor(memo.get())) {
        object = memo->get(object.get());
      }
      memo = top_context();
      auto parent = memo->getParent();
      if (parent) {
        object = parent->deep(object.get());
      }
      #if !USE_LAZY_DEEP_CLONE
      get();
      #endif
    }
  }

  WeakCOW(WeakCOW<Any> && o) = default;
  WeakCOW<Any>& operator=(const WeakCOW<Any>& o) = default;
  WeakCOW<Any>& operator=(WeakCOW<Any> && o) = default;

  Any* get() {
    #if USE_LAZY_DEEP_CLONE
    assert(memo->forwardPull() == top_context());
    auto forward = memo->forwardGet();
    if (forward != memo.get()) {
      object = forward->getParent()->deep(object.get());
      memo = forward;
    }
    object = memo->get(object.get());
    #endif
    return object.get();
  }

  Any* get() const {
    /* even in a const context, do want to update the pointer through lazy
     * deep clone mechanisms */
    return const_cast<WeakCOW<Any>*>(this)->get();
  }
  Any* pull() {
    #if USE_LAZY_DEEP_CLONE
    assert(memo->forwardPull() == top_context());
    auto forward = memo->forwardPull();
    if (forward != memo.get()) {
      object = forward->getParent()->deep(object.get());
      memo = forward;
    }
    object = memo->pull(object.get());
    #endif
    return object.get();
  }

  Any* pull() const {
    /* even in a const context, do want to update the pointer through lazy
     * deep clone mechanisms */
    return const_cast<WeakCOW<Any>*>(this)->pull();
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
  bool operator==(const WeakCOW<U>& o) const {
    return get() == o.get();
  }

  template<class U>
  bool operator!=(const WeakCOW<U>& o) const {
    return get() != o.get();
  }

  /**
   * Dynamic cast. Returns `nullptr` if unsuccessful.
   */
  template<class U>
  WeakCOW<U> dynamic_pointer_cast() const {
    return WeakCOW<U>(dynamic_cast<U*>(object.get()), memo.get());
  }

  /**
   * Static cast. Undefined if unsuccessful.
   */
  template<class U>
  WeakCOW<U> static_pointer_cast() const {
    return WeakCOW<U>(static_cast<U*>(object.get()), memo.get());
  }

protected:
  /**
   * The object.
   */
  WeakPtr<Any> object;

  /**
   * The memo.
   */
  ContextPtr memo;
};
}
