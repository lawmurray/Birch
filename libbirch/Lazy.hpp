/**
 * @file
 */
#pragma once

#include "libbirch/external.hpp"
#include "libbirch/type.hpp"
#include "libbirch/Any.hpp"
#include "libbirch/Shared.hpp"
#include "libbirch/Init.hpp"
#include "libbirch/Label.hpp"
#include "libbirch/LabelPtr.hpp"

namespace libbirch {
/**
 * Wrapper for a smart pointer type to apply lazy deep clone semantics.
 *
 * @ingroup libbirch
 *
 * @tparam P Pointer type, e.g. Shared or Init.
 */
template<class P>
class Lazy {
  template<class Q> friend class Lazy;
public:
  using value_type = typename P::value_type;
  using pointer_type = P;
  using label_type = Init<Label>;

  /**
   * Constructor.
   */
  Lazy(const std::nullptr_t& nil) {
    //
  }

  /**
   * Constructor.
   */
  Lazy(value_type* ptr, Label* label = nullptr) :
      object(ptr),
      label(label ? label : ptr->Any::getLabel()) {
    //
  }

  /**
   * Constructor with in-place construction of referent, using default
   * constructor.
   *
   * Allocates a new object of the type pointed to by this, and initializes
   * it by calling its default constructor.
   */
  Lazy() :
      object(new value_type()),
      label(root_label) {
    static_assert(std::is_default_constructible<value_type>::value,
        "invalid call to class constructor");
    // ^ ideally this condition would be checked with SFINAE, but the
    //   definition of value_type may not be available at the point that a
    //   pointer to it is declared, causing a compile error
  }

  /**
   * Constructor with in-place construction of referent, with single
   * argument.
   *
   * @tparam Arg Argument type.
   *
   * @param arg Argument.
   *
   * Allocates a new object of the type pointed to by this, and initializes
   * it by calling the constructor with the given argument.
   *
   * SFINAE ensures that the Lazy(const Q&) constructor is preferred over
   * this one when the argument is a pointer of the same or derived type.
   */
  template<class Arg, std::enable_if_t<!std::is_base_of<value_type,
      typename raw<Arg>::type>::value,int> = 0>
  explicit Lazy(const Arg& arg) :
      object(new value_type(arg)),
      label(root_label) {
    //
  }

  /**
   * Constructor with in-place construction of referent, with two or more
   * arguments.
   *
   * @tparam Arg1 First argument type.
   * @tparam Arg2 Second argument type.
   * @tparam Args... Argument types.
   *
   * @param arg1 First argument.
   * @param arg2 Second argument.
   * @param args... Arguments.
   *
   * Allocates a new object of the type pointed to by this, and initializes
   * it by calling its constructor with the given arguments.
   */
  template<class Arg1, class Arg2, class... Args>
  explicit Lazy(const Arg1& arg1, const Arg2& arg2, const Args&... args) :
      object(new value_type(arg1, arg2, args...)),
      label(root_label) {
    //
  }

  /**
   * Copy constructor.
   */
  Lazy(const Lazy& o) :
      object(o.get()),
      label(o.label) {
    // ^ o.get() maintains the single-reference optimization
  }

  /**
   * Generic copy constructor.
   */
  template<class Q, std::enable_if_t<std::is_base_of<value_type,
      typename Q::value_type>::value,int> = 0>
  Lazy(const Lazy<Q>& o) :
      object(o.get()),
      label(o.label) {
    // ^ o.get() maintains the single-reference optimization
  }

  /**
   * Move constructor.
   */
  Lazy(Lazy&&) = default;

  /**
   * Generic move constructor.
   */
  template<class Q, std::enable_if_t<std::is_base_of<value_type,
      typename Q::value_type>::value,int> = 0>
  Lazy(Lazy<Q>&& o) :
      object(std::move(o.object)),
      label(std::move(o.label)) {
    //
  }

  /**
   * Destructor.
   */
  ~Lazy() {
    //
  }

  /**
   * Correctly initialize after a bitwise copy.
   */
  void bitwiseFix(Label* newLabel) {
    object.bitwiseFix();
    new (&label) label_type(newLabel);  // overwrite with new label
    pullNoLock();
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
   * Value assignment.
   */
  template<class U, std::enable_if_t<is_value<U>::value,int> = 0>
  const Lazy<P>& operator=(const U& o) const {
    *get() = o;
    return *this;
  }

  /**
   * Copy assignment.
   */
  Lazy& operator=(const Lazy& o) {
    label = o.label;
    // ^ must go first, next line may invalidate o as a reference
    object = o.object;
    // ^ o.get() maintains the single-reference optimization
    return *this;
  }

  /**
   * Generic copy assignment.
   */
  template<class Q, std::enable_if_t<std::is_base_of<value_type,
      typename Q::value_type>::value,int> = 0>
  Lazy& operator=(const Lazy<Q>& o) {
    label = o.label;
    // ^ must go first, next line may invalidate o as a reference
    object = o.object;
    // ^ o.get() maintains the single-reference optimization
    return *this;
  }

  /**
   * Move assignment.
   */
  Lazy& operator=(Lazy&& o) {
    label = std::move(o.label);
    // ^ must go first, next line may invalidate o as a reference
    object = std::move(o.object);
    return *this;
  }

  /**
   * Generic move assignment.
   */
  template<class Q, std::enable_if_t<std::is_base_of<value_type,
      typename Q::value_type>::value,int> = 0>
  Lazy& operator=(Lazy<Q>&& o) {
    label = std::move(o.label);
    // ^ must go first, next line may invalidate o as a reference
    object = std::move(o.object);
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
    auto label = this->label.get();  // ensures only single read of atomic
    if (label) {
      return label->get(object);
    } else {
      assert(!object.query());
      return nullptr;
    }
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
    auto label = this->label.get();  // ensures only single read of atomic
    if (label) {
      return label->pull(object);
    } else {
      assert(!object.query());
      return nullptr;
    }
  }

  /**
   * Get the raw pointer for read-only use, without cloning.
   */
  value_type* pull() const {
    return const_cast<Lazy*>(this)->pull();
  }

  /**
   * Get the raw pointer for read-only use, without cloning.
   */
  value_type* pullNoLock() {
    auto label = this->label.get();  // ensures only single read of atomic
    if (label) {
      return label->pullNoLock(object);
    } else {
      assert(!object.query());
      return nullptr;
    }
  }

  /**
   * Dereference.
   */
  template<class Q = P>
  value_type& operator*() const {
    return *get();
  }

  /**
   * Member access.
   */
  template<class Q = P>
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

  /**
   * Finish.
   */
  void finish(Label* label) {
    if (getLabel() != label) {
      /* cross pointer, finish copies with get() */
      get()->Any::finish(label);
    } else {
      /* not a cross pointer, just pull() */
      pull()->Any::finish(label);
    }
  }

  /**
   * Freeze.
   */
  void freeze() {
    pull()->Any::freeze();
  }

  /**
   * Mark.
   */
  void mark() {
    object.mark();
    label.mark();
  }

  /**
   * Scan.
   */
  void scan() {
    object.scan();
    label.scan();
  }

  /**
   * Reach.
   */
  void reach() {
    object.reach();
    label.reach();
  }

  /**
   * Collect.
   */
  void collect() {
    object.collect();
    label.collect();
  }

private:
  /**
   * Object.
   */
  pointer_type object;

  /**
   * Label.
   */
  label_type label;
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
struct raw<Lazy<P>> {
  using type = typename raw<P>::type;
};

template<class P, unsigned N>
struct is_acyclic<Lazy<P>,N> {
  using pointer_type = typename Lazy<P>::pointer_type;
  using label_type = typename Lazy<P>::label_type;
  static const bool value = is_acyclic<label_type,N>::value &&
      is_acyclic<pointer_type,N>::value;
};

template<class T>
auto canonical(const Lazy<Shared<T>>& o) {
  return o;
}

template<class T>
auto canonical(Lazy<Shared<T>>&& o) {
  return std::move(o);
}

template<class T>
auto canonical(const Lazy<Init<T>>& o) {
  return Lazy<Shared<T>>(o);
}

template<class T, std::enable_if_t<std::is_base_of<Any,T>::value,int> = 0>
auto canonical(T* ptr) {
  return Lazy<Shared<T>>(ptr);
}

}
