/**
 * @file
 */
#pragma once

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
  using this_type = WeakCOW<T>;
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
  WeakCOW(T* object) :
      super_type(object) {
    //
  }

  /**
   * Constructor.
   */
  WeakCOW(const SharedPtr<T>& object) :
      super_type(object) {
    //
  }

  /**
   * Constructor.
   */
  WeakCOW(const WeakPtr<T>& object) :
      super_type(object) {
    //
  }

  /**
   * Constructor.
   */
  WeakCOW(const SharedCOW<T>& o) :
      super_type(o) {
    //
  }

  /**
   * Constructor.
   */
  WeakCOW(T* object, Memo* memo) :
      super_type(object, memo) {
    //
  }

  /**
   * Copy constructor.
   */
  WeakCOW(const WeakCOW<T>& o) = default;

  /**
   * Move constructor.
   */
  WeakCOW(WeakCOW<T>&& o) = default;

  /**
   * Copy assignment.
   */
  WeakCOW<T>& operator=(const WeakCOW<T>& o) = default;

  /**
   * Move assignment.
   */
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
   * Pull through generations.
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
  using this_type = WeakCOW<value_type>;
  using root_type = this_type;

  WeakCOW(const Nil& = nil) :
      object(),
      memo(fiberMemo) {
    assert(pullOnConstruct());
  }

  WeakCOW(Any* object) :
      object(object),
      memo(fiberMemo) {
    assert(pullOnConstruct());
  }

  WeakCOW(const SharedPtr<Any>& object) :
      object(object),
      memo(fiberMemo) {
    assert(pullOnConstruct());
  }

  WeakCOW(const WeakPtr<Any>& object) :
      object(object),
      memo(fiberMemo) {
    assert(pullOnConstruct());
  }

  WeakCOW(const SharedCOW<Any>& o) :
      object(o.object),
      memo(o.memo) {
    assert(pullOnConstruct());
  }

  WeakCOW(Any* object, Memo* memo) :
      object(object),
      memo(memo) {
    assert(pullOnConstruct());
  }

  WeakCOW(const WeakCOW<Any>& o) :
      object(fiberClone ? fiberMemo->deepPull(o.pull()) : o.object),
      memo(fiberClone ? fiberMemo : o.memo) {
    assert(pullOnConstruct());
  }

  WeakCOW(WeakCOW<Any> && o) = default;

  WeakCOW<Any>& operator=(const WeakCOW<Any>& o) = default;

  WeakCOW<Any>& operator=(WeakCOW<Any>&& o) = default;

  Any* pull() const {
    auto self = const_cast<WeakCOW<Any>*>(this);
    self->object = self->memo->pull(object.get());
    return object.get();
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

private:
  /**
   * On construction, all pointers should be correctly pulled forward to the
   * world on the pointer.
   */
  bool pullOnConstruct() const {
    return !object || memo->deepPull(object.get()) == memo->pull(object.get());
  }
};
}
