/**
 * @file
 */
#pragma once

#include "libbirch/global.hpp"
#include "libbirch/memory.hpp"
#include "libbirch/SharedPtr.hpp"
#include "libbirch/Memo.hpp"
#include "libbirch/Any.hpp"
#include "libbirch/Nil.hpp"

namespace bi {
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
  using this_type = SharedCOW<T>;
  using super_type = SharedCOW<typename super_type<T>::type>;
  using root_type = typename super_type::root_type;

  /**
   * Constructor.
   */
  SharedCOW(const Nil& = nil) :
      super_type() {
    //
  }

  /**
   * Constructor.
   */
  SharedCOW(T* object) :
      super_type(object) {
    //
  }

  /**
   * Constructor.
   */
  SharedCOW(const SharedPtr<T>& object) :
      super_type(object) {
    //
  }

  /**
   * Constructor.
   */
  SharedCOW(const WeakPtr<T>& object) :
      super_type(object) {
    //
  }

  /**
   * Constructor.
   */
  SharedCOW(const WeakCOW<T>& o);

  /**
   * Constructor.
   */
  SharedCOW(T* object, Memo* memo) :
      super_type(object, memo) {
    //
  }

  /**
   * Copy constructor.
   */
  SharedCOW(const SharedCOW<T>& o) = default;

  /**
   * Move constructor.
   */
  SharedCOW(SharedCOW<T>&& o) = default;

  /**
   * Copy assignment.
   */
  SharedCOW<T>& operator=(const SharedCOW<T>& o) = default;

  /**
   * Move assignment.
   */
  SharedCOW<T>& operator=(SharedCOW<T>&& o) = default;

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
  SharedCOW<T>& operator=(SharedCOW<U>&& o) {
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
   * Get the raw pointer.
   */
  T* get() const {
    return static_cast<T*>(root_type::get());
  }

  /**
   * Get the raw pointer while mapping, but not cloning. This is used as an
   * optimization for read-only access..
   */
  T* pull() const {
    return static_cast<T*>(root_type::pull());
  }

  /**
   * Lazy deep clone.
   */
  SharedCOW<T> clone() const {
    T* o = this->pull();
    Memo* m = construct<Memo>(this->memo);
    return SharedCOW<T>(o, m);
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
   * Call operator.
   */
  template<class ... Args>
  auto operator()(Args ... args) const {
    return (*get())(args...);
  }
};

template<>
class SharedCOW<Any> {
  template<class U> friend class WeakCOW;
public:
  using value_type = Any;
  using this_type = SharedCOW<value_type>;
  using root_type = this_type;

  SharedCOW(const Nil& = nil) :
      object(),
      memo(fiberMemo) {
    assert(invariant());
  }

  SharedCOW(Any* object) :
      object(object),
      memo(fiberMemo) {
    assert(invariant());
  }

  SharedCOW(const SharedPtr<Any>& object) :
      object(object),
      memo(fiberMemo) {
    assert(invariant());
  }

  SharedCOW(const WeakPtr<Any>& object) :
      object(object),
      memo(fiberMemo) {
    assert(invariant());
  }

  SharedCOW(Any* object, Memo* memo) :
      object(object),
      memo(memo) {
    assert(invariant());
  }

  SharedCOW(const WeakCOW<Any>& o);

  SharedCOW(const SharedCOW<Any>& o) :
      object(fiberClone ? fiberMemo->deepPull(o.pull()) : o.object),
      memo(fiberClone ? fiberMemo : o.memo) {
    assert(invariant());
  }

  SharedCOW(SharedCOW<Any> && o) = default;

  SharedCOW<Any>& operator=(const SharedCOW<Any>& o) = default;

  SharedCOW<Any>& operator=(SharedCOW<Any>&& o) = default;

  /**
   * Is the pointer not null?
   */
  bool query() const {
    return static_cast<bool>(object);
  }

  Any* get() const {
    /* despite the pointer being accessed in a const context, we do want to
     * update it through the copy-on-write mechanism for performance
     * reasons */
    auto self = const_cast<SharedCOW<Any>*>(this);
    self->object = self->memo->get(object.get());
    return object.get();
  }

  Any* pull() const {
    /* despite the pointer being accessed in a const context, we do want to
     * update it through the copy-on-write mechanism for performance
     * reasons */
    auto self = const_cast<SharedCOW<Any>*>(this);
    self->object = self->memo->pull(object.get());
    return object.get();
  }

  SharedCOW<Any> clone() const {
    Any* o = pull();
    Memo* m = bi::construct<Memo>(memo);
    return SharedCOW<Any>(o, m);
  }

  Memo* getMemo() const {
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
  SharedPtr<Memo> memo;

private:
  /**
   * Construction post-condition for copy-on-write implementation.
   */
  bool invariant() const {
    return !object || memo->deepPull(object.get()) == memo->pull(object.get());
  }
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
  assert(invariant());
}
