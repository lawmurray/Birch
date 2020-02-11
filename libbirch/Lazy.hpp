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
 * Wrapper for a smart pointer type to apply lazy deep clone semantics.
 *
 * @ingroup libbirch
 *
 * @tparam P Pointer type. Either SharedPtr, WeakPtr or InitPtr.
 */
template<class P, class Enable = void>
class Lazy : public Lazy<typename P::super_type> {
public:
  using pointer_type = P;
  using value_type = typename P::value_type;
  using super_type = Lazy<typename P::super_type>;
  using shared_type = Lazy<typename P::shared_type>;
  using weak_type = Lazy<typename P::weak_type>;
  using init_type = Lazy<typename P::init_type>;

  /**
   * Constructor.
   */
  Lazy(const Nil& = nil) :
      super_type() {
    //
  }

  /**
   * Constructor.
   */
  Lazy(Label* context, const Nil& = nil) :
      super_type(context) {
    //
  }

  /**
   * Constructor.
   */
  Lazy(Label* context, value_type* object, const bool cross = false) :
      super_type(context, object, cross) {
    //
  }

  /**
   * Constructor.
   */
  Lazy(Label* context, const P& object, const bool cross = false) :
      super_type(context, object, cross) {
    //
  }

  /**
   * Copy constructor.
   */
  Lazy(const shared_type& o) :
      super_type(o) {
    //
  }

  /**
   * Copy constructor.
   */
  Lazy(Label* context, const shared_type& o) :
      super_type(context, o) {
    //
  }

  /**
   * Copy constructor.
   */
  Lazy(const weak_type& o) :
      super_type(o) {
    //
  }

  /**
   * Copy constructor.
   */
  Lazy(Label* context, const weak_type& o) :
      super_type(context, o) {
    //
  }

  /**
   * Copy constructor.
   */
  Lazy(const init_type& o) :
      super_type(o) {
    //
  }

  /**
   * Copy constructor.
   */
  Lazy(Label* context, const init_type& o) :
      super_type(context, o) {
    //
  }

  /**
   * Move constructor.
   */
  Lazy(Label* context, Lazy<P>&& o) :
      super_type(context, o) {
    //
  }

  /**
   * Deep copy constructor.
   */
  Lazy(Label* context, Label* label, const Lazy<P>& o) :
      super_type(context, label, o) {
    //
  }

  /**
   * Copy assignment.
   */
  Lazy& assign(Label* context, const Lazy<P>& o) {
    super_type::assign(context, o);
    return *this;
  }

  /**
   * Move assignment.
   */
  Lazy& assign(Label* context, Lazy<P>&& o) {
    super_type::assign(context, o);
    return *this;
  }

  /**
   * Value assignment.
   */
  template<class U, typename = std::enable_if_t<is_value<U>::value>>
  Lazy& assign(Label* context, const U& o) {
    *get() = o;
    return *this;
  }

  /**
   * Value conversion.
   */
  template<class U, IS_CONVERTIBLE(value_type,U)>
  operator U() const {
    return get()->operator U();
  }

  /**
   * Get the raw pointer, with lazy cloning.
   */
  P get() {
    return super_type::get().template static_pointer_cast<P>();
  }

  /**
   * Get the raw pointer, with lazy cloning.
   */
  auto get() const {
    return const_cast<Lazy*>(this)->get();
  }

  /**
   * Get the raw pointer for read-only use, without cloning.
   */
  P pull() {
    return super_type::pull().template static_pointer_cast<P>();
  }

  /**
   * Get the raw pointer for read-only use, without cloning.
   */
  auto pull() const {
    return const_cast<Lazy*>(this)->pull();
  }

  /**
   * Start lazy deep clone.
   */
  Lazy clone(Label* context) const {
    return super_type::clone(context).template static_pointer_cast<P>(context);
  }

  /**
   * Dereference.
   */
  P& operator*() const {
    return *get();
  }

  /**
   * Member access.
   */
  P operator->() const {
    return get();
  }
};

/**
 * Wrapper for a smart pointer type to apply lazy deep clone semantics.
 *
 * @ingroup libbirch
 *
 * @tparam P Pointer type. Either SharedPtr, WeakPtr or InitPtr.
 */
template<class P>
class Lazy<P,std::enable_if_t<std::is_same<typename P::value_type,
    libbirch::Any>::value>> {
  template<class Q, class Enable1> friend class Lazy;
public:
  using pointer_type = P;
  using value_type = typename P::value_type;
  using shared_type = Lazy<typename P::shared_type>;
  using weak_type = Lazy<typename P::weak_type>;
  using init_type = Lazy<typename P::init_type>;

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
  Lazy(Label* context, value_type* object, const bool cross = false) :
      object(object),
      label(0),
      cross(false) {
    if (object) {
      setLabel(context, false);
    }
  }

  /**
   * Constructor.
   */
  Lazy(Label* context, const P& object, const bool cross = false) :
      object(object),
      label(0),
      cross(false) {
    if (object) {
      setLabel(context, false);
    }
  }

  /**
   * Copy constructor.
   */
  Lazy(const shared_type& o) :
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
  Lazy(Label* context, const shared_type& o) :
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
  Lazy(const weak_type& o) :
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
  Lazy(Label* context, const weak_type& o) :
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
  Lazy(const init_type& o) :
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
  Lazy(Label* context, const init_type& o) :
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
  Lazy(Lazy&& o) :
      object(std::move(o.object)),
      label(o.label),
      cross(o.cross) {
    o.label = 0;
    o.cross = false;
  }

  /**
   * Move constructor.
   */
  Lazy(Label* context, Lazy&& o) :
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
  Lazy(Label* context, Label* label, const Lazy& o) :
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
  Lazy<P>& assign(Label* context, const Lazy<P>& o) {
    if (o.query()) {
      replaceLabel(o.getLabel(), o.getLabel() != context);
      object = o.get();
    } else {
      release();
    }
    return *this;
  }

  /**
   * Move assignment.
   */
  Lazy<P>& assign(Label* context, Lazy<P>&& o) {
    if (o.query()) {
      replaceLabel(o.getLabel(), o.getLabel() != context);
      object = std::move(o.object);
    } else {
      release();
    }
    return *this;
  }

  /**
   * Release the pointer.
   */
  void release() {
    releaseLabel();
    object.release();
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
    getLabel()->get(object);
    return object;
  }

  /**
   * Get the raw pointer, with lazy cloning.
   */
  auto get() const {
    return const_cast<Lazy*>(this)->get();
  }

  /**
   * Get the raw pointer for read-only use, without cloning.
   */
  P& pull() {
    getLabel()->pull(object);
    return object;
  }

  /**
   * Get the raw pointer for read-only use, without cloning.
   */
  auto pull() const {
    return const_cast<Lazy*>(this)->pull();
  }

  /**
   * Start lazy deep clone.
   */
  Lazy clone(Label* context) const {
    assert(object);
    pull();
    startFreeze();
    return Lazy(getLabel()->fork(), object, true);
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
    return const_cast<Lazy*>(this)->startFreeze();
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
    return const_cast<Lazy*>(this)->freeze();
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
    return const_cast<Lazy*>(this)->thaw(label);
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
    return const_cast<Lazy*>(this)->startFinish();
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
    return const_cast<Lazy*>(this)->finish();
  }

  /**
   * Dereference.
   */
  P& operator*() const {
    return *get();
  }

  /**
   * Member access.
   */
  P operator->() const {
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
    this->label = 0;
    this->cross = false;
    if (label && cross) {
      label->decShared();
    }
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
