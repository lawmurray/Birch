/**
 * @file
 */
#pragma once

#include "libbirch/external.hpp"
#include "libbirch/type.hpp"
#include "libbirch/Any.hpp"
#include "libbirch/Label.hpp"
#include "libbirch/InitPtr.hpp"

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
  Lazy(const Q& ptr, Label* label = rootLabel) :
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
      label(rootLabel) {
    //
  }

  /**
   * Constructor.
   */
  template<class Q, std::enable_if_t<is_base_of<Any*,Q>::value,int> = 0>
  Lazy(const Q& ptr, Label* label = rootLabel) :
      object(ptr),
      label(label) {
    //
  }

  /**
   * Copy constructor.
   */
  Lazy(const Lazy& o) :
      object(o.object),
      label(o.label) {
    //
  }

  /**
   * Move constructor.
   */
  Lazy(Lazy&& o) :
      object(std::move(o.object)),
      label(o.label) {
    //
  }

  /**
   * Destructor.
   */
  ~Lazy() {
    //
  }

  /**
   * Copy assignment.
   */
  Lazy& operator=(const Lazy& o) {
    if (o.query()) {
      object.replace(o.get());
      label = o.label;
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
      object = std::move(o.object);
      label = std::move(o.label);
    } else {
      release();
    }
    return *this;
  }

  /**
   * Release the pointer.
   */
  void release() {
    object.release();
    label.release();
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
    label->get(object);
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
    label->pull(object);
    return object.get();
  }

  /**
   * Get the raw pointer for read-only use, without cloning.
   */
  Any* pull() const {
    return const_cast<Lazy*>(this)->pull();
  }

  /**
   * Get the label associated with the pointer.
   */
  Label* getLabel() const {
    return label.get();
  }

  /**
   * Set the label associated with the pointer.
   */
  void setLabel(Label* label) {
    this->label.replace(label);
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
   * Object.
   */
  P object;

  /**
   * Label.
   */
  InitPtr<Label> label;
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
