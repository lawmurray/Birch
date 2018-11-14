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
   * Deep clone.
   */
  SharedCOW<T> clone() const {
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
    //
  }

  SharedCOW(Any* object) :
      object(object),
      memo(fiberMemo) {
    //
  }

  SharedCOW(const SharedPtr<Any>& object) :
      object(object),
      memo(fiberMemo) {
    //
  }

  SharedCOW(const WeakPtr<Any>& object) :
      object(object),
      memo(fiberMemo) {
    //
  }

  SharedCOW(Any* object, Memo* memo) :
      object(object),
      memo(memo) {
    //
  }

  SharedCOW(const WeakCOW<Any>& o);

  SharedCOW(const SharedCOW<Any>& o) :
      object((fiberClone && o.object) ? o.pull()->deepPull(fiberMemo) : o.object),
      memo(fiberClone ? fiberMemo : o.memo) {
    //
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
    if (object) {
      auto self = const_cast<SharedCOW<Any>*>(this);
      self->object = object->get(self->memo);
    }
    return object.get();
  }

  Any* pull() const {
    /* despite the pointer being accessed in a const context, we do want to
     * update it through the copy-on-write mechanism for performance
     * reasons */
    if (object) {
      auto self = const_cast<SharedCOW<Any>*>(this);
      self->object = object->pull(self->memo);
    }
    return object.get();
  }

  SharedCOW<Any> clone() const {
    return SharedCOW<Any>(get()->clone(), memo->clone());
  }

  Memo* getMemo() const {
    return memo;
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
    return SharedCOW<U>(dynamic_cast<U*>(object.get()), memo);
  }

  /**
   * Static cast. Undefined if unsuccessful.
   */
  template<class U>
  SharedCOW<U> static_pointer_cast() const {
    return SharedCOW<U>(static_cast<U*>(object.get()), memo);
  }

protected:
  /**
   * The object.
   */
  SharedPtr<Any> object;

  /**
   * The memo.
   */
  Memo* memo;
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
