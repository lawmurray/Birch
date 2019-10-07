/**
 * @file
 */
#pragma once
#if ENABLE_LAZY_DEEP_CLONE

#include "libbirch/LazyAny.hpp"
#include "libbirch/LazyLabel.hpp"
#include "libbirch/Nil.hpp"
#include "libbirch/LabelPtr.hpp"
#include "libbirch/thread.hpp"

namespace libbirch {
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
  using value_type = typename P::value_type;

  /**
   * Constructor.
   */
  LazyPtr(const Nil& = nil){
    //
  }

  /**
   * Constructor.
   */
  LazyPtr(Label* label, value_type* object) :
      label(object ? label : nullptr),
      object(object) {
    //
  }

  /**
   * Constructor.
   */
  LazyPtr(Label* label, const P& object) :
      label(object ? label : nullptr),
      object(object) {
    //
  }

  /**
   * Deep copy constructor.
   */
  LazyPtr(Label* label, const LazyPtr<P>& o) :
      label(label),
      object() {
    if (o.object) {
      if (o.isCross()) {
        o.finish();
        o.freeze();
      }
      object = o.object;
    }
  }

  /**
   * Copy constructor.
   */
  LazyPtr<P>(const LazyPtr<P>& o) :
      label(o.label),
      object(o.get()) {
    //
  }

  /**
   * Generic copy constructor.
   */
  template<class Q, typename = std::enable_if_t<std::is_base_of<value_type,
      typename Q::value_type>::value>>
  LazyPtr<P>(const LazyPtr<Q>& o) :
      label(o.label),
      object(o.get()) {
    //
  }

  /**
   * Move constructor.
   */
  LazyPtr<P>(LazyPtr<P>&& o) :
      label(std::move(o.label)),
      object(std::move(o.get())) {
    //
  }

  /**
   * Generic move constructor.
   */
  template<class Q, typename = std::enable_if_t<std::is_base_of<value_type,
      typename Q::value_type>::value>>
  LazyPtr<P>(LazyPtr<Q>&& o) :
      label(std::move(o.label)),
      object(std::move(o.get())) {
    //
  }

  /**
   * Copy assignment.
   */
  LazyPtr<P>& operator=(const LazyPtr<P>& o) {
    label = o.label;
    object = o.get();
    return *this;
  }

  /**
   * Move assignment.
   */
  LazyPtr<P>& operator=(LazyPtr<P>&& o) {
    label = std::move(o.label);
    object = std::move(o.get());
    return *this;
  }

  /**
   * Nil assignment.
   */
  LazyPtr<P>& operator=(const Nil&) {
    object.release();
    label.release();
    return *this;
  }

  /**
   * Nullptr assignment.
   */
  LazyPtr<P>& operator=(const std::nullptr_t&) {
    object.release();
    label.release();
    return *this;
  }

  /**
   * Value assignment.
   */
  template<class U/*, typename = std::enable_if_t<std::is_assignable<value_type,U>::value>*/>
  LazyPtr<P>& operator=(const U& o) {
    *get() = o;
    return *this;
  }

  /**
   * Value conversion.
   */
  template<class U, typename = std::enable_if_t<std::is_convertible<value_type,U>::value>>
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
  auto& get() {
    auto raw = object.get();
    if (raw && raw->isFrozen()) {
      raw = static_cast<value_type*>(label->get(raw));
      object.replace(raw);
    }
    return object;
  }

  /**
   * Get the raw pointer, with lazy cloning.
   */
  auto& get() const {
    return const_cast<LazyPtr<P>*>(this)->get();
  }

  /**
   * Get the raw pointer for read-only use, without cloning.
   */
  auto& pull() {
    auto raw = object.get();
    if (raw && raw->isFrozen()) {
      raw = static_cast<value_type*>(label->pull(raw));
      object.replace(raw);
    }
    return object;
  }

  /**
   * Get the raw pointer for read-only use, without cloning.
   */
  auto& pull() const {
    return const_cast<LazyPtr<P>*>(this)->pull();
  }

  /**
   * Deep clone.
   */
  LazyPtr<P> clone(Label* label) const {
    assert(object);
    pull();
    startFreeze();
    return LazyPtr<P>(label->fork(), object);
  }

  /**
   * Start a freeze operation. This is just like freeze(), but manages thread
   * safety on entry and exit.
   */
  void startFreeze() {
    if (object) {
      freezeLock.enter();
      freeze();
      freezeLock.exit();
    }
  }

  /**
   * Start a freeze operation. This is just like freeze(), but manages thread
   * safety on entry and exit.
   */
  void startFreeze() const {
    return const_cast<LazyPtr<P>*>(this)->startFreeze();
  }

  /**
   * Freeze.
   */
  void freeze() {
    if (object) {
      object->freeze();
      label->freeze();
    }
  }

  /**
   * Freeze.
   */
  void freeze() const {
    return const_cast<LazyPtr<P>*>(this)->freeze();
  }

  /**
   * Thaw.
   */
  void thaw(LazyLabel* label) {
    if (isCross()) {
      startFinish();
      startFreeze();
    }
    if (object) {
      this->label.replace(label);
    }
  }

  /**
   * Thaw.
   */
  void thaw(LazyLabel* label) const {
    return const_cast<LazyPtr<P>*>(this)->thaw(label);
  }

  /**
   * Start a finish operation. This is just like finish(), but manages thread
   * safety on entry and exit.
   */
  void startFinish() {
    if (object) {
      finishLock.enter();
      finish();
      finishLock.exit();
    }
  }

  /**
   * Start a freeze operation. This is just like finish(), but manages thread
   * safety on entry and exit.
   */
  void startFinish() const {
    return const_cast<LazyPtr<P>*>(this)->startFinish();
  }

  /**
   * Finish.
   */
  void finish() {
    if (object) {
      get();
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
    return label.isCross();
  }

  /**
   * Dereference.
   */
  value_type& operator*() const {
    return *get();
  }

  /**
   * Member access.
   */
  value_type* operator->() const {
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
   * Dynamic cast.
   */
  template<class U>
  auto dynamic_pointer_cast() const {
    auto cast = object.template dynamic_pointer_cast<U>();
    return LazyPtr<decltype(cast)>(label.get(), cast);
  }

  /**
   * Static cast.
   */
  template<class U>
  auto static_pointer_cast() const {
    auto cast = object.template static_pointer_cast<U>();
    return LazyPtr<decltype(cast)>(label.get(), cast);
  }

private:
  /**
   * Label of the object.
   */
  LabelPtr label;

  /**
   * Object.
   */
  P object;
};
}

#endif
