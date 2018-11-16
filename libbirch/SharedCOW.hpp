/**
 * @file
 */
#pragma once

#include "libbirch/global.hpp"
#include "libbirch/class.hpp"
#include "libbirch/memory.hpp"
#include "libbirch/SharedPtr.hpp"
#include "libbirch/InitPtr.hpp"
#include "libbirch/Memo.hpp"
#include "libbirch/Any.hpp"
#include "libbirch/Enter.hpp"
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
  SharedCOW(SharedCOW<T> && o) = default;

  /**
   * Copy assignment.
   */
  SharedCOW<T>& operator=(const SharedCOW<T>& o) = default;

  /**
   * Move assignment.
   */
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
  T* map() {
    return static_cast<T*>(root_type::map());
  }

  /**
   * Map the raw pointer, without lazy cloning.
   */
  T* map() const {
    return static_cast<T*>(root_type::map());
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

  SharedCOW(const Nil& = nil) {
    //
  }

  SharedCOW(Any* object) :
      object(object) {
    //
  }

  SharedCOW(const SharedPtr<Any>& object) :
      object(object) {
    //
  }

  SharedCOW(const WeakPtr<Any>& object) :
      object(object) {
    //
  }

  SharedCOW(Any* object, Memo* memo) :
      object(object),
      memo(memo) {
    //
  }

  SharedCOW(const WeakCOW<Any>& o);

  SharedCOW(const SharedCOW<Any>& o) :
      object(o.object),
      memo(o.memo) {
    if (cloneMemo && object) {
      object = object->deepPull(memo);
      memo = cloneMemo;
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
    if (object) {
      object = object->get(memo);
      memo = nullptr;
    }
    return object.get();
  }

  Any* get() const {
    if (object) {
      return object->get(memo);
    } else {
      return object.get();
    }
  }

  Any* map() {
    if (object) {
      object = object->map(memo);
    }
    return object.get();
  }

  Any* map() const {
    if (object) {
      return object->map(memo);
    } else {
      return object.get();
    }
  }

  SharedCOW<Any> clone() const {
    auto memo1 = make_object<Memo>(memo);
    Enter enter(memo1);
    auto object1 = get()->clone();
    object1->setMemo(memo1);
    return SharedCOW<Any>(object1, memo1);
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
  InitPtr<Memo> memo;
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
