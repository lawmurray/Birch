/**
 * @file
 */
#pragma once

#include "libbirch/Any.hpp"
#include "libbirch/Atomic.hpp"
#include "libbirch/type.hpp"
#include "libbirch/memory.hpp"

namespace libbirch {
/**
 * Shared object.
 *
 * @ingroup libbirch
 *
 * @tparam T Type, must derive from Any.
 * 
 * Supports reference counted garbage collection. Cycles are collected by
 * periodically calling collect(); currently this must be done manually at
 * opportune times.
 * 
 * @attention While Shared maintains a pointer to a referent object, it likes
 * to pretend it's not a pointer. This behavior differs from
 * [`std::shared_ptr`](https://en.cppreference.com/w/cpp/memory/shared_ptr).
 * In particular, its default constructor does not initialize the pointer to
 * `nullptr`, but rather default-constructs an object of type `T` and sets the
 * pointer to that. Consider using a [`std::optional`](https://en.cppreference.com/w/cpp/utility/optional/optional)
 * with a Shared value instead of `nullptr`.
 */
template<class T>
class Shared {
  template<class U> friend class Shared;
  friend class Marker;
  friend class Scanner;
  friend class Reacher;
  friend class Collector;
  friend class Spanner;
  friend class Bridger;
  friend class Copier;
public:
  using value_type = T;

  /**
   * Default constructor. Constructs a new referent using the default
   * constructor.
   */
  Shared() :
      Shared(new T(), false, false) {
    //
  }

  /**
   * Constructor. Constructs a new referent with the given arguments. The
   * first is a placeholder (pass [`std::in_place`](https://en.cppreference.com/w/cpp/utility/in_place))
   * to distinguish this constructor from copy and move constructors.
   * 
   * @note [`std::optional`](https://en.cppreference.com/w/cpp/utility/optional/)
   * behaves similarly with regard to [`std::in_place`](https://en.cppreference.com/w/cpp/utility/in_place).
   */
  template<class... Args>
  Shared(std::in_place_t, Args&&... args) :
      Shared(new T(std::forward<Args>(args)...), false, false) {
    //
  }

  /**
   * Constructor.
   * 
   * @param ptr Raw pointer.
   */
  Shared(T* ptr, const bool b = false, const bool c = false) :
      ptr(ptr),
      b(b),
      c(c) {
    if (ptr) {
      ptr->incShared();
    }
  }


  /**
   * Copy constructor.
   */
  Shared(const Shared& o) :
      Shared(o.get(), o.c, false) {
    //
  }

  /**
   * Generic copy constructor.
   */
  template<class U, std::enable_if_t<std::is_base_of<T,U>::value,int> = 0>
  Shared(const Shared<U>& o) :
      Shared(static_cast<T*>(o.get()), o.c, false) {
    //
  }

  /**
   * Move constructor.
   */
  Shared(Shared&& o) :
      ptr(o.ptr.exchange(nullptr)), b(o.b), c(o.c) {
    o.b = false;
    o.c = false;
  }

  /**
   * Generic move constructor.
   */
  template<class U, std::enable_if_t<std::is_base_of<T,U>::value,int> = 0>
  Shared(Shared<U>&& o) :
      ptr(o.ptr.exchange(nullptr)), b(o.b), c(o.c) {
    o.b = false;
    o.c = false;
  }

  /**
   * Destructor.
   */
  ~Shared() {
    release();
  }

  /**
   * Copy assignment.
   */
  Shared& operator=(const Shared& o) {
    replace(o.get());
    b = false;
    c = false;
    return *this;
  }

  /**
   * Generic copy assignment.
   */
  template<class U, std::enable_if_t<std::is_base_of<T,U>::value,int> = 0>
  Shared& operator=(const Shared<U>& o) {
    replace(o.get());
    b = false;
    c = false;
    return *this;
  }

  /**
   * Move assignment.
   */
  Shared& operator=(Shared&& o) {
    auto ptr = o.ptr.exchange(nullptr);
    auto old = this->ptr.exchange(ptr);
    if (old) {
      if (ptr == old) {
        old->decSharedReachable();
      } else {
        old->decShared();
      }
    }
    std::swap(b, o.b);
    std::swap(c, o.c);
    return *this;
  }

  /**
   * Generic move assignment.
   */
  template<class U, std::enable_if_t<std::is_base_of<T,U>::value,int> = 0>
  Shared& operator=(Shared<U>&& o) {
    auto ptr = o.ptr.exchange(nullptr);
    auto old = this->ptr.exchange(ptr);
    if (old) {
      if (ptr == old) {
        old->decSharedReachable();
      } else {
        old->decShared();
      }
    }
    std::swap(b, o.b);
    std::swap(c, o.c);
    return *this;
  }

  /**
   * Value assignment.
   */
  template<class U, std::enable_if_t<is_value<U>::value,int> = 0>
  Shared<T>& operator=(const U& o) {
    *get() = o;
    return *this;
  }

  /**
   * Value assignment.
   */
  template<class U, std::enable_if_t<is_value<U>::value,int> = 0>
  const Shared<T>& operator=(const U& o) const {
    *get() = o;
    return *this;
  }

  /**
   * Is the pointer not null?
   *
   * This is used instead of an `operator bool()` so as not to conflict with
   * conversion operators in the referent type.
   */
  bool query() const {
    return ptr.load() != nullptr;
  }

  /**
   * Get the raw pointer.
   */
  T* get();

  /**
   * Get the raw pointer.
   */
  T* get() const {
    return const_cast<Shared<T>*>(this)->get();
  }

  /**
   * Get the raw pointer as const.
   */
  const T* read() const {
    return get();
  }

  /**
   * Deep copy.
   */
  Shared<T> copy();

  /**
   * Deep copy.
   */
  Shared<T> copy() const {
    return const_cast<Shared<T>*>(this)->copy();
  }

  /**
   * Replace. Sets the raw pointer to a new value and returns the previous
   * value.
   */
  T* replace(T* ptr) {
    if (ptr) {
      ptr->incShared();
    }
    auto old = this->ptr.exchange(ptr);
    if (old) {
      if (ptr == old) {
        old->decSharedReachable();
      } else {
        old->decShared();
      }
    }
    b = false;
    c = false;
    return old;
  }

  /**
   * Release. Sets the raw pointer to null and returns the previous value.
   */
  T* release() {
    auto old = ptr.exchange(nullptr);
    if (old) {
      old->decShared();
    }
    b = false;
    c = false;
    return old;
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
   * Call on referent.
   */
  template<class... Args>
  auto& operator()(Args&&... args) {
    return (*get())(std::forward<Args>(args)...);
  }

private:
  /**
   * Raw pointer.
   */
  Atomic<T*> ptr;

  /**
   * Is this a bridge?
   */
  bool b;

  /**
   * If this is a bridge, is it a far bridge?
   */
  bool c;
};

template<class T>
struct is_value<Shared<T>> {
  static const bool value = false;
};

template<class T>
struct is_pointer<Shared<T>> {
  static const bool value = true;
};

template<class T, int N>
struct is_acyclic<Shared<T>,N> {
  // because pointers are polymorphic, the class must be both final and
  // acyclic for the pointer to be considered acyclic
  static const bool value = std::is_final<T>::value && is_acyclic_class<T,N-1>::value;
};

template<class T>
struct is_acyclic<Shared<T>,0> {
  static const bool value = false;
};
}

#include "libbirch/Spanner.hpp"
#include "libbirch/Bridger.hpp"
#include "libbirch/Copier.hpp"

template<class T>
T* libbirch::Shared<T>::get() {
  T* v = ptr.load();
  if (b && !c) {
    b = false;
    v = static_cast<T*>(Copier().visit(static_cast<Any*>(v)));
    replace(v);
  }
  return v;
}

template<class T>
libbirch::Shared<T> libbirch::Shared<T>::copy() {
  /* find bridges */
  Spanner().visit(0, 1, *this);
  Bridger().visit(1, 0, *this);

  Any* v = ptr.load();
  if (b) {
    /* if a bridge, then both source and copy can be bridges */
    c = false;
    return Shared<T>(static_cast<T*>(v), true, false);
  } else {
    /* otherwise copy up to the next bridges */
    return Shared<T>(static_cast<T*>(Copier().visit(v)), false, false);
  }
}
