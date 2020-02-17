/**
 * @file
 */
#pragma once

#include "libbirch/external.hpp"
#include "libbirch/type.hpp"
#include "libbirch/Any.hpp"
#include "libbirch/Label.hpp"
#include "libbirch/Nil.hpp"
#include "libbirch/SharedPtr.hpp"
#include "libbirch/WeakPtr.hpp"
#include "libbirch/InitPtr.hpp"
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
  using value_type = typename P::value_type;
  using this_type = Lazy<P>;
  using super_type = Lazy<typename P::super_type>;

  /**
   * Constructor.
   */
  Lazy(const std::nullptr_t& nil) :
      super_type(nil) {
    //
  }

  /**
   * Constructor.
   */
  template<class Q, std::enable_if_t<is_base_of<P,Q>::value,int> = 0>
  Lazy(const Q& ptr, Label* label = nullptr) :
      super_type(ptr, label) {
    //
  }

  /**
   * Constructor with in-place construction of referent.
   *
   * Allocates a new object of the type pointed to by this, and initializes
   * it by calling its default constructor.
   */
  Lazy() : super_type(new value_type()) {
    static_assert(std::is_default_constructible<value_type>::value,
        "invalid call to class constructor");
    // ^ ideally this condition would be checked with SFINAE, but the
    //   definition of value_type may not be available at the point that a
    //   pointer to it is declared, causing a compile error
  }

  /**
   * Constructor with in-place construction of referent.
   *
   * @tparam Args... Argument types.
   *
   * @param args... Arguments.
   *
   * Allocates a new object of the type pointed to by this, and initializes
   * it by calling its constructor with the given arguments.
   *
   * SFINAE insures that the Lazy(const Q&) constructor is preferred over
   * this one when the argument is a pointer of the same or type. Note that
   * in the Birch language it is not possible for a class to have a
   * constructor that would accept such an argument anyway.
   */
  template<class Arg, class... Args, std::enable_if_t<!is_base_of<P,Arg>::value,int> = 0>
  explicit Lazy(Arg arg, Args... args) : super_type(new value_type(arg, args...)) {
    static_assert(std::is_constructible<value_type,Arg,Args...>::value,
        "invalid call to class constructor");
    // ^ ideally this condition would be checked with SFINAE, but the
    //   definition of value_type may not be available at the point that a
    //   pointer to it is declared, causing a compile error
  }

  /**
   * Value assignment.
   */
  template<class U, std::enable_if_t<is_value<U>::value && std::is_assignable<value_type,U>::value,int> = 0>
  Lazy& operator=(const U& o) {
    *get() = o;
    return *this;
  }

  /**
   * Value conversion.
   */
  template<class U, std::enable_if_t<is_value<U>::value && std::is_convertible<value_type,U>::value,int> = 0>
  operator U() const {
    return get()->operator U();
  }

  /**
   * Get the raw pointer, with lazy cloning.
   */
  value_type* get() {
    return static_cast<value_type*>(super_type::get());
  }

  /**
   * Get the raw pointer, with lazy cloning.
   */
  value_type* get() const {
    return static_cast<value_type*>(super_type::get());
  }

  /**
   * Get the raw pointer for read-only use, without cloning.
   */
  value_type* pull() {
    return static_cast<value_type*>(super_type::pull());
  }

  /**
   * Get the raw pointer for read-only use, without cloning.
   */
  value_type* pull() const {
    return static_cast<value_type*>(super_type::pull());
  }

  /**
   * Start lazy deep clone.
   */
  Lazy clone() const {
    pull();
    this->startFreeze();
    return Lazy(get(), this->getLabel()->fork());
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
};

/**
 * Wrapper for a smart pointer type to apply lazy deep clone semantics.
 *
 * @ingroup libbirch
 *
 * @tparam P Pointer type. Either SharedPtr, WeakPtr or InitPtr.
 */
template<class P>
class Lazy<P,std::enable_if_t<std::is_same<typename P::value_type,libbirch::Any>::value>> {
  template<class Q, class Enable1> friend class Lazy;
public:
  using value_type = typename P::value_type;
  using this_type = Lazy<Any>;

  /**
   * Constructor.
   */
  Lazy(const std::nullptr_t& nil) :
      object(nullptr),
      label(0),
      cross(0) {
    //
  }

  /**
   * Constructor.
   */
  template<class Q, std::enable_if_t<is_base_of<Any*,Q>::value,int> = 0>
  Lazy(const Q& ptr, Label* label = nullptr) :
      object(ptr),
      label(0),
      cross(0) {
    setLabel(label);
  }

  /**
   * Copy constructor.
   */
  Lazy(const Lazy& o) :
      object(o.object),
      label(o.label),
      cross(0) {
    //
  }

  /**
   * Move constructor.
   */
  Lazy(Lazy&& o) :
      object(std::move(o.object)),
      label(o.label),
      cross(o.cross) {
    //
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
  Lazy& operator=(const Lazy& o) {
    if (o.query()) {
      replaceLabel(o.getLabel());
      object.replace(o.get());
    } else {
      release();
    }
    return *this;
  }

  /**
   * Move assignment.
   */
  Lazy& operator=(Lazy&& o) {
    if (o.query()) {
      replaceLabel(o.getLabel());
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
   *
   * This is used instead of an `operator bool()` so as not to conflict with
   * conversion operators in the referent type.
   */
  bool query() const {
    return object.query();
  }

  /**
   * Get the raw pointer, with lazy cloning.
   */
  Any* get() {
    getLabel()->get(object);
    return object.get();
  }

  /**
   * Get the raw pointer, with lazy cloning.
   */
  Any* get() const {
    return const_cast<Lazy*>(this)->get();
  }

  /**
   * Get the raw pointer for read-only use, without cloning.
   */
  Any* pull() {
    getLabel()->pull(object);
    return object.get();
  }

  /**
   * Get the raw pointer for read-only use, without cloning.
   */
  Any* pull() const {
    return const_cast<Lazy*>(this)->pull();
  }

  /**
   * Start a freeze operation. This is just like freeze(), but manages thread
   * safety on entry and exit.
   */
  void startFreeze() {
    if (object.query()) {
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
    if (object.query()) {
      ///@todo
      //object->freeze();
      //getLabel()->freeze();
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
    if (object.query()) {
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
      ///@todo
      //object->finish();
    }
  }

  /**
   * Finish.
   */
  void finish() const {
    return const_cast<Lazy*>(this)->finish();
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

private:
  /**
   * Set the label.
   */
  void setLabel(Label* label) {
    assert(this->label == 0);
    assert(!this->cross);
    this->label = reinterpret_cast<intptr_t>(label);
    this->cross = label != nullptr;
    if (cross) {
      label->incShared();
    }
  }

  /**
   * Replace the label.
   */
  void replaceLabel(Label* label) {
    auto oldLabel = this->getLabel();
    auto oldCross = this->isCross();
    this->label = reinterpret_cast<intptr_t>(label);
    this->cross = label != nullptr;
    if (cross) {
      label->incShared();
    }
    if (oldCross) {
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
    if (cross) {
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

template<class P>
struct raw_type<Lazy<P>> {
  using type = typename raw_type<P>::type;
};
}
