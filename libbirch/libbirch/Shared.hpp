/**
 * @file
 */
#pragma once

#include "libbirch/Atomic.hpp"
#include "libbirch/type.hpp"
#include "libbirch/memory.hpp"

namespace libbirch {
/**
 * %Shared object.
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
  friend class BiconnectedCollector;
  friend class Spanner;
  friend class Bridger;
  friend class Copier;
  friend class BiconnectedCopier;
  friend class Destroyer;
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
      ptr->incShared_();
    }
  }

  /**
   * Copy constructor.
   */
  Shared(const Shared& o) : ptr(o.ptr), b(o.b) {
    if (ptr) {
      if (biconnected_copy()) {
        if (b) {
          unpack(ptr)->incShared_();
        } else {
          // deferred until Copier or BiconnectedCopier visits and updates
        }
      } else {
        if (b) {
          store(o.get());  // copy next biconnected component
          b = false;
        }
        unpack(ptr)->incShared_();
      }
    }
  }

  /**
   * Generic copy constructor.
   */
  template<class U, std::enable_if_t<std::is_base_of<T,U>::value,int> = 0>
  Shared(const Shared<U>& o) :
      ptr(pack(o.get())),
      b(false) {
    if (ptr) {
      unpack(ptr)->incShared_();
    }
  }

  /**
   * Move constructor.
   */
  Shared(Shared&& o) : ptr(o.ptr), b(o.b) {
    o.store(nullptr);
    o.b = false;
  }

  /**
   * Generic move constructor.
   */
  template<class U, std::enable_if_t<std::is_base_of<T,U>::value,int> = 0>
  Shared(Shared<U>&& o) : ptr(o.ptr), b(o.b) {
    o.store(nullptr);
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
    replace(o);
    return *this;
  }

  /**
   * Generic copy assignment.
   */
  template<class U, std::enable_if_t<std::is_base_of<T,U>::value,int> = 0>
  Shared& operator=(const Shared<U>& o) {
    replace(o);
    return *this;
  }

  /**
   * Move assignment.
   */
  Shared& operator=(Shared&& o) {
    replace(std::move(o));
    return *this;
  }

  /**
   * Generic move assignment.
   */
  template<class U, std::enable_if_t<std::is_base_of<T,U>::value,int> = 0>
  Shared& operator=(Shared<U>&& o) {
    replace(std::move(o));
    return *this;
  }

  /**
   * Value assignment.
   */
  template<class U, std::enable_if_t<!std::is_base_of<T,U>::value,int> = 0>
  Shared<T>& operator=(const U& o) {
    *get() = o;
    return *this;
  }

  /**
   * Value assignment.
   */
  template<class U, std::enable_if_t<!std::is_base_of<T,U>::value,int> = 0>
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
    return ptr != 0;
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
   * Bridge finding. Finds bridges in the reachable graph. Must be followed by
   * copy() for correct behavior.
   */
  void bridge();

  /**
   * Bridge finding.
   */
  void bridge() const {
    return const_cast<Shared<T>*>(this)->bridge();
  }

  /**
   * Deep copy. Copies the subgraph up to the nearest reachable bridges, the
   * remainder deferred until needed. If bridge() has been called previously,
   * this constitutes a lazy copy, otherwise an eager copy. It is correct
   * either way.
   */
  Shared<T> copy();

  /**
   * Deep copy.
   */
  Shared<T> copy() const {
    return const_cast<Shared<T>*>(this)->copy();
  }

  /**
   * Replace.
   */
  template<class U>
  void replace(const Shared<U>& o) {
    auto ptr = unpack(this->ptr);
    auto b = this->b;

    /* when duplicating a pointer, may need to trigger a biconnected copy */
    auto ptr1 = o.get();
    if (ptr1) {
      ptr1->incShared_();
    }

    this->ptr = pack(ptr1);
    this->b = false;  // if it was a bridge, now copied

    if (ptr) {
      if (ptr == ptr1) {
        ptr->decSharedReachable_();
      } else if (b) {
        ptr->decSharedBridge_();
      } else {
        ptr->decShared_();
      }
    }
  }

  /**
   * Replace.
   */
  template<class U>
  void replace(Shared<U>&& o) {
    auto ptr = unpack(this->ptr);
    auto b = this->b;

    this->ptr = o.ptr;
    this->b = o.b;

    if (ptr) {
      if (ptr == unpack(this->ptr)) {
        ptr->decSharedReachable_();
      } else if (b) {
        ptr->decSharedBridge_();
      } else {
        ptr->decShared_();
      }
    }

    o.ptr = 0;
    o.b = false;
  }

  /**
   * Release the referent.
   */
  void release() {
    auto ptr = unpack(this->ptr);
    auto b = this->b;

    this->ptr = 0;
    this->b = false;

    if (ptr) {
      if (b) {
        ptr->decSharedBridge_();
      } else {
        ptr->decShared_();
      }
    }
  }

  /**
   * @internal
   * 
   * Release the referent, during collection of a biconnected component.
   */
  void releaseBiconnected() {
    auto ptr = unpack(this->ptr);
    auto b = this->b;

    this->ptr = 0;
    this->b = false;

    if (ptr) {
      if (b) {
        ptr->decSharedBridge_();
      } else {
        ptr->decSharedBiconnected_();
      }
    }
  }

  /**
   * Dereference.
   */
  T& operator*() {
    return *get();
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
  T* operator->() {
    return get();
  }

  /**
   * Member access.
   */
  T* operator->() const {
    return get();
  }

  /**
   * Equality operator.
   */
  template<class U>
  bool operator==(const Shared<U>& o) const {
    return get() == o.get();
  }

  /**
   * Non-equality operator.
   */
  template<class U>
  bool operator!=(const Shared<U>& o) const {
    return get() != o.get();
  }

  /**
   * Call on referent.
   */
  template<class... Args>
  auto& operator()(Args&&... args) {
    return (*get())(std::forward<Args>(args)...);
  }

  /**
   * Call on referent.
   */
  template<class... Args>
  auto& operator()(Args&&... args) const {
    return const_cast<Shared<T>*>(this)->operator()(std::forward<Args>(args)...);
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
  static T* unpack(int64_t ptr) {
    return reinterpret_cast<T*>(ptr);
  }

  /**
   * Pack a raw pointer before storing.
   */
  static int64_t pack(T* ptr) {
    return reinterpret_cast<int64_t>(ptr);
  }

  /**
   * Raw pointer.
   */
  int64_t ptr:63;

  /**
   * Is this a bridge edge?
   */
  bool b:1;
};

template<class T>
struct is_pointer<Shared<T>> {
  static const bool value = true;
};

template<class T>
struct unwrap_pointer<Shared<T>> {
  using type = T;
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
    if (!o->isUniqueHead_()) {  // last reference optimization
      /* copy biconnected component */
      assert(!biconnected_copy());
      biconnected_copy(true);
      assert(biconnected_copy());
      o = BiconnectedCopier(o).visitObject(o);
      biconnected_copy(true);
      assert(!biconnected_copy());

      /* replace pointer */
      o->incShared_();
      auto old = unpack(ptr);
      ptr = pack(o);
      old->decSharedBridge_();
    }
    b = false;  // no longer a bridge
  }
  return o;
}

template<class T>
void libbirch::Shared<T>::bridge() {
  Spanner().visit(0, 1, *this);
  Bridger().visit(1, 0, *this);
}

template<class T>
libbirch::Shared<T> libbirch::Shared<T>::copy() {
  T* o = unpack(ptr);
  if (!b) {
    /* the copy is *not* of a biconnected component here, use the
     * general-purpose Copier for this */
    assert(!biconnected_copy());
    biconnected_copy(true);
    assert(biconnected_copy());
    o = Copier().visitObject(o);
    biconnected_copy(true);
    assert(!biconnected_copy());
  }
  return Shared<T>(o, b);
}
