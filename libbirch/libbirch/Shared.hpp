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
      ptr(pack(ptr)),
      b(b) {
    if (ptr) {
      ptr->incShared();
    }
  }


  /**
   * Copy constructor.
   */
  Shared(const Shared& o) : ptr(o.ptr), b(o.b) {
    if (ptr) {
      if (biconnected_copy()) {
        if (b) {
          unpack(ptr)->incShared();
        } else {
          // deferred until Copier or BiconnectedCopier visits and updates
        }
      } else {
        if (b) {
          store(o.get());  // copy next biconnected component
          b = false;
        }
        unpack(ptr)->incShared();
      }
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
  Shared(Shared&& o) : ptr(o.ptr), b(o.b) {
    o.ptr = pack(nullptr);
    o.b = false;
  }

  /**
   * Generic move constructor.
   */
  template<class U, std::enable_if_t<std::is_base_of<T,U>::value,int> = 0>
  Shared(Shared<U>&& o) : ptr(o.ptr), b(o.b) {
    o.ptr = pack(nullptr);
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
    return *this;
  }

  /**
   * Generic copy assignment.
   */
  template<class U, std::enable_if_t<std::is_base_of<T,U>::value,int> = 0>
  Shared& operator=(const Shared<U>& o) {
    replace(o.get());
    return *this;
  }

  /**
   * Move assignment.
   */
  Shared& operator=(Shared&& o) {
    exchange(std::move(o));
    return *this;
  }

  /**
   * Generic move assignment.
   */
  template<class U, std::enable_if_t<std::is_base_of<T,U>::value,int> = 0>
  Shared& operator=(Shared<U>&& o) {
    exchange(std::move(o));
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
    return unpack(ptr) != nullptr;
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
   * Get the raw pointer without copy-on-use.
   */
  T* load() const {
    return unpack(ptr);
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
   * Exchange with another.
   */
  template<class U>
  void exchange(Shared<U>&& o) {
    auto old = ptr;
    ptr = o.ptr;
    b = o.b;
    o.ptr = pack(nullptr);
    o.b = false;
    if (old) {
      if (ptr == old) {
        unpack(old)->decSharedReachable();
      } else {
        unpack(old)->decShared();
      }
    }
  }

  /**
   * Replace. Sets the raw pointer to a new value and returns the previous
   * value.
   */
  void replace(T* ptr) {
    if (ptr) {
      ptr->incShared();
    }
    auto old = unpack(this->ptr);
    this->ptr = pack(ptr);
    this->b = false;
    if (old) {
      if (ptr == old) {
        old->decSharedReachable();
      } else {
        old->decShared();
      }
    }
  }

  /**
   * Release. Sets the raw pointer to null and returns the previous value.
   */
  void release() {
    auto old = ptr;
    ptr = pack(nullptr);
    b = false;
    if (old) {
      unpack(old)->decShared();
    }
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
   * Store the raw pointer without reference count updates.
   */
  void store(T* ptr) {
    this->ptr = pack(ptr);
  }

  /**
   * Unpack a raw pointer after loading.
   */
  static T* unpack(intptr_t ptr) {
    return reinterpret_cast<T*>(ptr);
  }

  /**
   * Pack a raw pointer before storing.
   */
  static intptr_t pack(T* ptr) {
    return reinterpret_cast<intptr_t>(ptr);
  }

  /**
   * Raw pointer.
   */
  intptr_t ptr:(8*sizeof(intptr_t) - 1);

  /**
   * Is this a bridge?
   */
  bool b:1;
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
  T* o = unpack(ptr);
  if (b) {
    if (o->numShared() > 1) {  // no need to copy for last reference
      /* the copy is of a biconnected component here, used the optimized
       * BiconnectedCopier for this */
      assert(!biconnected_copy());
      biconnected_copy(true);
      assert(biconnected_copy());
      o = static_cast<T*>(BiconnectedCopier(o).visit(static_cast<Any*>(o)));
      biconnected_copy(true);
      assert(!biconnected_copy());
      replace(o);
    }
    b = false;
  }
  return o;
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
  T* o = unpack(ptr);
  if (!b) {
    /* the copy is *not* of a biconnected component here, use the
     * general-purpose Copier for this */
    assert(!biconnected_copy());
    biconnected_copy(true);
    assert(biconnected_copy());
    o = static_cast<T*>(Copier().visit(static_cast<Any*>(o)));
    biconnected_copy(true);
    assert(!biconnected_copy());
  }
  return Shared<T>(o, b);
}
