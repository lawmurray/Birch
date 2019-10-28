/**
 * @file
 */
#pragma once

#include "libbirch/external.hpp"
#include "libbirch/type.hpp"
#include "libbirch/Any.hpp"
#include "libbirch/Label.hpp"
#include "libbirch/Nil.hpp"
#include "libbirch/thread.hpp"

namespace libbirch {
/**
 * Wraps a pointer type to apply lazy deep clone semantics.
 *
 * @ingroup libbirch
 *
 * @tparam P Pointer type. Either SharedPtr, WeakPtr or InitPtr.
 */
template<class P>
class Lazy {
  template<class U> friend class Lazy;
public:
  using pointer_type = P;
  using value_type = typename P::value_type;

  Lazy& operator=(const Lazy&) = delete;
  Lazy& operator=(Lazy&&) = delete;

  /**
   * Constructor.
   */
  Lazy(const Nil& = nil) :
      object(),
      label(0),
      cross(false) {
    //
  }

  /**
   * Constructor.
   */
  Lazy(Label* context, const Nil& = nil) :
      object(),
      label(0),
      cross(false) {
    //
  }

  /**
   * Constructor.
   */
  Lazy(Label* context, value_type* object) :
      object(object),
      label(reinterpret_cast<intptr_t>(context)),
      cross(false) {
    //
  }

  /**
   * Constructor.
   */
  Lazy(Label* context, const P& object) :
      object(object),
      label(reinterpret_cast<intptr_t>(context)),
      cross(false) {
    //
  }

  /**
   * Copy constructor.
   */
  Lazy(const Lazy<P>& o) :
      object(o.get()),
      label(o.label),
      cross(o.cross) {
    if (isCross()) {
      getLabel()->incShared();
    }
  }

  /**
   * Copy constructor.
   */
  template<class Q, IS_CONVERTIBLE(Q,P)>
  Lazy(const Lazy<Q>& o) :
      object(o.get()),
      label(o.label),
      cross(o.cross) {
    if (isCross()) {
      getLabel()->incShared();
    }
  }

  /**
   * Copy constructor.
   */
  Lazy(Label* context, const Lazy<P>& o) :
      object(o.get()),
      label(0),
      cross(false) {
    if (object) {
      setLabel(o.getLabel(), o.getLabel() != context);
    }
  }

  /**
   * Copy constructor.
   */
  template<class Q, IS_CONVERTIBLE(Q,P)>
  Lazy(Label* context, const Lazy<Q>& o) :
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
  Lazy(Lazy<P>&& o) :
      object(std::move(o.object)),
      label(o.label),
      cross(o.cross) {
    o.label = 0;
    o.cross = false;
  }

  /**
   * Move constructor.
   */
  template<class Q, IS_CONVERTIBLE(Q,P)>
  Lazy(Lazy<Q>&& o) :
      object(std::move(o.object)),
      label(o.label),
      cross(o.cross) {
    o.label = 0;
    o.cross = false;
  }

  /**
   * Move constructor.
   */
  Lazy(Label* context, Lazy<P>&& o) :
      object(std::move(o.object)),
      label(0),
      cross(false) {
    if (object) {
      setLabel(o.getLabel(), o.getLabel() != context);
    }
  }

  /**
   * Move constructor.
   */
  template<class Q, IS_CONVERTIBLE(Q,P)>
  Lazy(Label* context, Lazy<Q>&& o) :
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
  Lazy(Label* context, Label* label, const Lazy<P>& o) :
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
  ~Lazy() {
    releaseLabel();
  }

  /**
   * Copy assignment.
   */
  template<class Q>
  Lazy& assign(Label* context, const Lazy<Q>& o) {
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
  template<class Q>
  Lazy& assign(Label* context, Lazy<Q>&& o) {
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
  Lazy<P>& assign(Label* context, const U& o) {
    *get() = o;
    return *this;
  }

  /**
   * Release the pointer.
   */
  void release() {
    object.release();
    releaseLabel();
  }

  /**
   * Value conversion.
   */
  template<class U, IS_CONVERTIBLE(value_type,U)>
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
  P& get() {
    auto raw = object.get();
    if (raw && raw->isFrozen()) {
      raw = static_cast<value_type*>(getLabel()->get(raw));
      object.replace(raw);
      assert(object);
    }
    return object;
  }

  /**
   * Get the raw pointer, with lazy cloning.
   */
  auto get() const {
    return const_cast<Lazy<P>*>(this)->get();
  }

  /**
   * Get the raw pointer for read-only use, without cloning.
   */
  P& pull() {
    auto raw = object.get();
    if (raw && raw->isFrozen()) {
      raw = static_cast<value_type*>(getLabel()->pull(raw));
      object.replace(raw);
      assert(object);
    }
    return object;
  }

  /**
   * Get the raw pointer for read-only use, without cloning.
   */
  auto pull() const {
    return const_cast<Lazy<P>*>(this)->pull();
  }

  /**
   * Start lazy deep clone.
   */
  Lazy<P> clone(Label* context) const {
    assert(object);
    pull();
    startFreeze();
    return Lazy<P>(context, getLabel()->fork(), object);
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
    return const_cast<Lazy<P>*>(this)->startFreeze();
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
    return const_cast<Lazy<P>*>(this)->freeze();
  }

  /**
   * Thaw.
   */
  void thaw(Label* label) {
    if (isCross()) {
      startFinish();
      startFreeze();
    }
    if (object) {
      this->label = reinterpret_cast<intptr_t>(label);
    }
  }

  /**
   * Thaw.
   */
  void thaw(Label* label) const {
    return const_cast<Lazy<P>*>(this)->thaw(label);
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
    return const_cast<Lazy<P>*>(this)->startFinish();
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
    return const_cast<Lazy<P>*>(this)->finish();
  }

  /**
   * Dereference.
   */
  auto& operator*() const {
    return *get();
  }

  /**
   * Member access.
   */
  auto operator->() const {
    return get();
  }

  /**
   * Equal comparison.
   */
  template<class U>
  bool operator==(const Lazy<U>& o) const {
    return get() == o.get();
  }

  /**
   * Not equal comparison.
   */
  template<class U>
  bool operator!=(const Lazy<U>& o) const {
    return get() != o.get();
  }

  /**
   * Dynamic cast.
   */
  template<class U>
  auto dynamic_pointer_cast(Label* context) const {
    auto cast = get().template dynamic_pointer_cast<typename U::pointer_type>();
    return Lazy<decltype(cast)>(getLabel(), cast);
  }

  /**
   * Static cast.
   */
  template<class U>
  auto static_pointer_cast(Label* context) const {
    auto cast = get().template static_pointer_cast<typename U::pointer_type>();
    return Lazy<decltype(cast)>(getLabel(), cast);
  }

private:
  /**
   * Constructor.
   */
  Lazy(Label* context, Label* label, const P& object) :
      object(object),
      label(0),
      cross(false) {
    if (object) {
      setLabel(label, label != context);
    }
  }

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

template<class P>
struct is_value<Lazy<P>> {
  static const bool value = false;
};

template<class P>
struct is_pointer<Lazy<P>> {
  static const bool value = true;
};
}
