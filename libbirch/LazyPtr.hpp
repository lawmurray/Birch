/**
 * @file
 */
#pragma once
#if ENABLE_LAZY_DEEP_CLONE

#include "libbirch/LazyAny.hpp"
#include "libbirch/LazyLabel.hpp"
#include "libbirch/Nil.hpp"
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
  using pointer_type = P;
  using value_type = typename P::value_type;

  LazyPtr& operator=(const LazyPtr&) = delete;
  LazyPtr& operator=(LazyPtr&&) = delete;

  /**
   * Constructor.
   */
  LazyPtr(const Nil& = nil) :
    object(),
    label(0),
    cross(false) {
    //
  }

  /**
   * Constructor.
   */
  LazyPtr(Label* context, const Nil& = nil) :
      object(),
      label(0),
      cross(false) {
    //
  }

  /**
   * Constructor.
   */
  LazyPtr(Label* context, const P& object) :
      object(object),
      label(reinterpret_cast<intptr_t>(context)),
      cross(false) {
    //
  }

  /**
   * Constructor.
   */
  LazyPtr(Label* context, const value_type* object) :
      object(object),
      label(reinterpret_cast<intptr_t>(context)),
      cross(false) {
    //
  }

  /**
   * Copy constructor.
   */
  LazyPtr(const LazyPtr<P>& o) :
      object(o.object),
      label(o.label),
      cross(false) {
    //
  }

  /**
   * Copy constructor.
   */
  template<class Q>
  LazyPtr(const LazyPtr<Q>& o) :
      object(o.object),
      label(o.label),
      cross(false) {
    //
  }

  /**
   * Copy constructor.
   */
  template<class Q>
  LazyPtr(Label* context, const LazyPtr<Q>& o) :
      object(o.get()),
      label(0),
      cross(false) {
    if (object) {
      setLabel(o.getLabel(), o.getLabel() != context);
    }
  }

  /**
   * Move constructor.
   */
  LazyPtr(LazyPtr<P>&& o) :
      object(std::move(o.object)),
      label(o.label),
      cross(o.cross) {
    o.label = 0;
    o.cross = false;
  }

  /**
   * Move constructor.
   */
  template<class Q>
  LazyPtr(LazyPtr<Q>&& o) :
      object(std::move(o.object)),
      label(o.label),
      cross(o.cross) {
    o.label = 0;
    o.cross = false;
  }

  /**
   * Move constructor.
   */
  template<class Q>
  LazyPtr(Label* context, LazyPtr<Q>&& o) :
      object(std::move(o.object)),
      label(0),
      cross(false) {
    if (object) {
      setLabel(o.getLabel(), o.getLabel() != context);
    }
  }

  /**
   * Deep copy constructor.
   */
  LazyPtr(Label* context, Label* label, const LazyPtr<P>& o) :
      object(),
      label(0),
      cross(false) {
    assert(context == label);
    if (o.object) {
      if (o.isCross()) {
        o.finish();
        o.freeze();
      }
      object = o.object;
      setLabel(label, false);
    }
  }

  /**
   * Destructor.
   */
  ~LazyPtr() {
    releaseLabel();
  }

  /**
   * Copy assignment.
   */
  LazyPtr& assign(Label* context, const LazyPtr<P>& o) {
    object = o.get();
    if (object) {
      replaceLabel(o.getLabel(), o.getLabel() != context);
    } else {
      releaseLabel();
    }
    return *this;
  }

  /**
   * Move assignment.
   */
  LazyPtr& assign(Label* context, LazyPtr<P>&& o) {
    object = std::move(o.get());
    if (object) {
      replaceLabel(o.getLabel(), o.getLabel() != context);
    } else {
      releaseLabel();
    }
    return *this;
  }

  /**
   * Value assignment.
   */
  template<class U, typename = std::enable_if_t<is_value<U>::value>>
  LazyPtr<P>& assign(Label* context, const U& o) {
    *get() = o;
    return *this;
  }

  /**
   * Value conversion.
   */
  template<class U, typename = std::enable_if_t<std::is_convertible<value_type,U>::value>>
  explicit operator U() const {
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
  value_type* get() {
    value_type* raw = object.get();
    if (raw && raw->isFrozen()) {
      raw = static_cast<value_type*>(getLabel()->get(raw));
      object.replace(raw);
    }
    return raw;
  }

  /**
   * Get the raw pointer, with lazy cloning.
   */
  auto get() const {
    return const_cast<LazyPtr<P>*>(this)->get();
  }

  /**
   * Get the raw pointer for read-only use, without cloning.
   */
  value_type* pull() {
    value_type* raw = object.get();
    if (raw && raw->isFrozen()) {
      raw = static_cast<value_type*>(getLabel()->pull(raw));
      object.replace(raw);
    }
    return raw;
  }

  /**
   * Get the raw pointer for read-only use, without cloning.
   */
  auto pull() const {
    return const_cast<LazyPtr<P>*>(this)->pull();
  }

  /**
   * Start lazy deep clone.
   */
  LazyPtr<P> clone(Label* context) const {
    assert(object);
    pull();
    startFreeze();
    return LazyPtr<P>(context, getLabel()->fork(), object);
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
      getLabel()->freeze();
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
  auto dynamic_pointer_cast(Label* context) const {
    auto cast = object.template dynamic_pointer_cast<typename U::pointer_type>();
    return LazyPtr<decltype(cast)>(getLabel(), cast);
  }

  /**
   * Static cast.
   */
  template<class U>
  auto static_pointer_cast(Label* context) const {
    auto cast = object.template static_pointer_cast<typename U::pointer_type>();
    return LazyPtr<decltype(cast)>(getLabel(), cast);
  }

private:
  /**
   * Get the label.
   */
  Label* getLabel() const {
    return reinterpret_cast<Label*>(this->label);
  }

  /**
   * Is this pointer crossed? A crossed pointer is to a context different to
   * that of the context in which it was created (e.g. the context of the
   * object to which it belongs).
   */
  bool isCross() const {
    return cross;
  }

  /**
   * Set the label.
   */
  void setLabel(Label* label, bool cross) {
    assert(this->label == 0);
    assert(!this->cross);
    this->label = reinterpret_cast<intptr_t>(label);
    this->cross = cross;
    if (label && cross) {
      label->incShared();
    }
  }

  /**
   * Replace the label.
   */
  void replaceLabel(Label* label, bool cross) {
    auto oldLabel = this->getLabel();
    auto oldCross = this->isCross();
    this->label = reinterpret_cast<intptr_t>(label);
    this->cross = cross;
    if (label && cross) {
      label->incShared();
    }
    if (oldLabel && oldCross) {
      oldLabel->decShared();
    }
  }

  /**
   * Release the label.
   */
  void releaseLabel() {
    auto label = getLabel();
    auto cross = isCross();
    if (label && cross) {
      label->decShared();
    }
    this->label = 0;
    this->cross = false;
  }

  /**
   * Object.
   */
  P object;

  /**
   * Raw pointer.
   */
  intptr_t label:63;

  /**
   * Is this pointer crossed? A crossed pointer is to a context different to
   * that of the context in which it was created (e.g. the context of the
   * object to which it belongs).
   */
  bool cross:1;
};
}

#endif
