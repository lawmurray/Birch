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
 * @tparam P Pointer type, e.g. a SharedPtr, WeakPtr, InitPtr.
 */
template<class P>
class Lazy {
  template<class Q> friend class Lazy;
public:
  using value_type = typename P::value_type;

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
  template<class Q, std::enable_if_t<is_base_of<P,Q>::value,int> = 0>
  Lazy(const Q& ptr, Label* label = rootLabel) :
      object(ptr),
      label(label) {
    //
  }

  /**
   * Generic copy constructor.
   */
  template<class Q, std::enable_if_t<is_base_of<P,Q>::value,int> = 0>
  Lazy(const Lazy<Q>& o) :
      object(o.object),
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
   * Constructor with in-place construction of referent.
   *
   * Allocates a new object of the type pointed to by this, and initializes
   * it by calling its default constructor.
   */
  Lazy() :
      object(new value_type()),
      label(rootLabel) {
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
  explicit Lazy(Arg arg, Args... args) :
      object(new value_type(arg, args...)),
      label(rootLabel) {
    static_assert(std::is_constructible<value_type,Arg,Args...>::value,
        "invalid call to class constructor");
    // ^ ideally this condition would be checked with SFINAE, but the
    //   definition of value_type may not be available at the point that a
    //   pointer to it is declared, causing a compile error
  }

  /**
   * Value assignment.
   */
  template<class U, std::enable_if_t<is_value<U>::value,int> = 0>
  Lazy<P>& operator=(const U& o) {
    *get() = o;
    return *this;
  }

  /**
   * Is the pointer not null?
   */
  bool query() const {
    return object.query();
  }

  /**
   * Get the raw pointer, with lazy cloning.
   */
  value_type* get() {
    label->get(object);
    return object.get();
  }

  /**
   * Get the raw pointer, with lazy cloning.
   */
  value_type* get() const {
    return const_cast<Lazy*>(this)->get();
  }

  /**
   * Get the raw pointer for read-only use, without cloning.
   */
  value_type* pull() {
    label->pull(object);
    return object.get();
  }

  /**
   * Get the raw pointer for read-only use, without cloning.
   */
  value_type* pull() const {
    return const_cast<Lazy*>(this)->pull();
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
