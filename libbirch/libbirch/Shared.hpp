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
  friend class BiconnectedCopier;
public:
  using value_type = T;

  /**
   * Default constructor. Constructs a new referent using the default
   * constructor.
   */
  Shared() :
      Shared(new T(), false) {
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
      Shared(new T(std::forward<Args>(args)...), false) {
    //
  }

  /**
   * Constructor.
   * 
   * @param ptr Raw pointer.
   * @param b Is this a bridge?
   */
  Shared(T* ptr, const bool b = false) :
      ptr(ptr),
      b(b) {
    if (ptr) {
      ptr->incShared();
    }
  }


  /**
   * Copy constructor.
   */
  Shared(const Shared& o) : ptr(nullptr), b(false) {
    if (o.b) {
      if (biconnected_copy()) {
        replace(o.load());
        b = true;
      } else {
        replace(o.get());
        b = false;
      }
    } else {
      replace(o.load());
    }
  }

  /**
   * Generic copy constructor.
   */
  template<class U, std::enable_if_t<std::is_base_of<T,U>::value,int> = 0>
  Shared(const Shared<U>& o) :
      Shared(o.get(), false) {
    //
  }

  /**
   * Move constructor.
   */
  Shared(Shared&& o) :
      ptr(o.ptr), b(o.b) {
    o.ptr = nullptr;
    o.b = false;
  }

  /**
   * Generic move constructor.
   */
  template<class U, std::enable_if_t<std::is_base_of<T,U>::value,int> = 0>
  Shared(Shared<U>&& o) :
      ptr(o.ptr), b(o.b) {
    o.ptr = nullptr;
    o.b = false;
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
    return *this;
  }

  /**
   * Generic copy assignment.
   */
  template<class U, std::enable_if_t<std::is_base_of<T,U>::value,int> = 0>
  Shared& operator=(const Shared<U>& o) {
    replace(o.get());
    b = false;
    return *this;
  }

  /**
   * Move assignment.
   */
  Shared& operator=(Shared&& o) {
    auto old = ptr;
    ptr = o.ptr;
    b = o.b;
    o.ptr = nullptr;
    o.b = false;
    if (old) {
      if (ptr == old) {
        old->decSharedReachable();
      } else {
        old->decShared();
      }
    }
    return *this;
  }

  /**
   * Generic move assignment.
   */
  template<class U, std::enable_if_t<std::is_base_of<T,U>::value,int> = 0>
  Shared& operator=(Shared<U>&& o) {
    auto old = ptr;
    ptr = o.ptr;
    b = o.b;
    o.ptr = nullptr;
    o.b = false;
    if (old) {
      if (ptr == old) {
        old->decSharedReachable();
      } else {
        old->decShared();
      }
    }
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
    return load() != nullptr;
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
   * Deep copy. Finds bridges in the reachable graph then immediately copies
   * the subgraph up to the nearest reachable bridges, the remainder deferred
   * until needed.
   */
  Shared<T> copy();

  /**
   * Deep copy.
   */
  Shared<T> copy() const {
    return const_cast<Shared<T>*>(this)->copy();
  }

  /**
   * Deep copy. Copies the subgraph up to the nearest reachable bridges, the
   * remainder deferred until needed. Unlike #copy(), does not attempt to find
   * new bridges, but does use existing bridges. This is suitable if eager
   * copying, rather than lazy copying, is preferred, or for the second and
   * subsequent copies when replicating a graph multiple times, when the
   * bridge finding has already been completed by the first copy (using
   * #copy()).
   */
  Shared<T> copy2();

  /**
   * Deep copy.
   */
  Shared<T> copy2() const {
    return const_cast<Shared<T>*>(this)->copy2();
  }

  /**
   * Replace. Sets the raw pointer to a new value and returns the previous
   * value.
   */
  T* replace(T* ptr) {
    if (ptr) {
      ptr->incShared();
    }
    auto old = this->ptr;
    this->ptr = ptr;
    this->b = false;
    if (old) {
      if (ptr == old) {
        old->decSharedReachable();
      } else {
        old->decShared();
      }
    }
    return old;
  }

  /**
   * Release. Sets the raw pointer to null and returns the previous value.
   */
  T* release() {
    auto old = ptr;
    ptr = nullptr;
    b = false;
    if (old) {
      old->decShared();
    }
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
   * Load the raw pointer as-is. Does not trigger copy-on-write.
   */
  T* load() const {
    return ptr;
  }

  /**
   * Store the raw pointer as-is. Does not update reference counts.
   */
  void store(T* o) {
    ptr = o;
  }

  /**
   * Raw pointer.
   */
  T* ptr;

  /**
   * Is this a bridge?
   */
  bool b;
};

template<class T>
struct is_value<Shared<T>> {
  static const bool value = false;
};

template<class T>
struct is_pointer<Shared<T>> {
  static const bool value = true;
};
}

#include "libbirch/Spanner.hpp"
#include "libbirch/Bridger.hpp"
#include "libbirch/Copier.hpp"
#include "libbirch/BiconnectedCopier.hpp"

template<class T>
T* libbirch::Shared<T>::get() {
  T* v = load();
  if (b) {
    b = false;
    if (v->numShared() > 1) {  // no need to copy for last reference
      /* the copy is of a biconnected component here, used the optimized
       * BiconnectedCopier for this */
      v = static_cast<T*>(BiconnectedCopier(v).visit(static_cast<Any*>(v)));
      replace(v);
    }
  }
  return v;
}

template<class T>
libbirch::Shared<T> libbirch::Shared<T>::copy() {
  /* find bridges */
  Spanner().visit(0, 1, *this);
  Bridger().visit(1, 0, *this);

  /* copy */
  return copy2();
}

template<class T>
libbirch::Shared<T> libbirch::Shared<T>::copy2() {
  Any* u = load();
  if (b) {
    return Shared<T>(static_cast<T*>(u), true);
  } else {
    /* the copy is *not* of a biconnected component here, use the
     * general-purpose Copier for this */
    return Shared<T>(static_cast<T*>(Copier().visit(u)), false);
  }
}
