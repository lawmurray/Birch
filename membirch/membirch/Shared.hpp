/**
 * @file
 */
#pragma once

#include "membirch/Atomic.hpp"
#include "membirch/type.hpp"
#include "membirch/memory.hpp"

namespace membirch {
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
  template<class U = T, std::enable_if_t<std::is_constructible_v<U>,int> = 0>
  Shared() :
      Shared(new T(), false) {
    //
  }

  /**
   * Constructor. Constructs a new referent using a constructor taking the
   * given arguments.
   */
  template<class Arg, class... Args, std::enable_if_t<
      !std::is_same_v<Shared<T>,std::decay_t<Arg>>,int> = 0>
  explicit Shared(Arg&& arg, Args&&... args) :
      Shared(new T(std::forward<Arg>(arg), std::forward<Args>(args)...), false) {
    // avoid use of std::is_constructible_v<T,Args...> in SFINAE here, as
    // clang++ giving compile errors (as of version 14.0.0)
  }

  /**
   * Constructor.
   * 
   * @param ptr Raw pointer.
   * @param bridge Is this a bridge?
   */
  Shared(T* ptr, const bool bridge = false) {
    if (ptr) {
      ptr->incShared_();
    }
    pack(ptr, bridge);
  }

  /**
   * Constructor.
   */
  Shared(std::nullptr_t) {
    pack(nullptr, false);
  }

  /**
   * Copy constructor.
   */
  Shared(const Shared& o) {
    auto [ptr, bridge] = o.unpack();
    if (ptr) {
      if (in_copy()) {
        if (bridge) {
          /* don't copy next biconnected component */
          ptr->incShared_();
        } else {
          // deferred until Copier or BiconnectedCopier visits and updates
        }
      } else {
        if (bridge) {
          /* copy next biconnected component */
          ptr = o.get();
          bridge = false;
        }
        ptr->incShared_();
      }
    }
    pack(ptr, bridge);
  }

  /**
   * Copy constructor.
   * 
   * This overload exists to disambiguate with Shared(Args&&...).
   */
  Shared(Shared& o) : Shared(const_cast<const Shared&>(o)) {
    //
  }

  /**
   * Generic copy constructor.
   */
  template<class U, std::enable_if_t<std::is_base_of_v<T,U>,int> = 0>
  Shared(const Shared<U>& o) {
    auto [ptr, bridge] = o.unpack();
    if (ptr) {
      if (bridge) {
        /* copy next biconnected component */
        ptr = o.get();
        bridge = false;
      }
      ptr->incShared_();
    }
    pack(ptr, bridge);
  }

  /**
   * Generic copy constructor.
   * 
   * This overload exists to disambiguate with Shared(Args&&...).
   */
  template<class U>
  Shared(Shared<U>& o) : Shared(const_cast<const Shared<U>&>(o)) {
    //
  }

  /**
   * Move constructor.
   */
  Shared(Shared&& o) :
      packed(o.packed.exchange(0)) {
    //
  }

  /**
   * Generic move constructor.
   */
  template<class U, std::enable_if_t<std::is_base_of_v<T,U>,int> = 0>
  Shared(Shared<U>&& o) :
      packed(o.packed.exchange(0)) {
    //
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
   * Get the raw pointer without copy-on-use.
   */
  T* load() const {
    return packed.load() & POINTER;
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
    auto [old, bridge] = unpack();

    /* when duplicating a pointer, may need to trigger a biconnected copy */
    auto ptr = o.get();
    if (ptr) {
      ptr->incShared_();
    }
    pack(ptr, false);  // if it was a bridge, now copied

    if (old) {
      if (old == ptr) {
        old->decSharedReachable_();
      } else if (bridge) {
        old->decSharedBridge_();
      } else {
        old->decShared_();
      }
    }
  }

  /**
   * Replace.
   */
  template<class U>
  void replace(Shared<U>&& o) {
    auto packed1 = o.packed.exchange(0);
    auto ptr = (U*)(packed1 & POINTER);
    auto [old, bridge] = unpack(packed.exchange(packed1));
    if (old) {
      if (old == ptr) {
        old->decSharedReachable_();
      } else if (bridge) {
        old->decSharedBridge_();
      } else {
        old->decShared_();
      }
    }
  }

  /**
   * Release the referent.
   */
  void release() {
    auto [old, bridge] = unpack(packed.exchange(0));
    if (old) {
      if (bridge) {
        old->decSharedBridge_();
      } else {
        old->decShared_();
      }
    }
  }

  /**
   * @internal
   * 
   * Release the referent, during collection of a biconnected component.
   */
  void releaseBiconnected() {
    auto [old, bridge] = unpack(packed.exchange(0));
    if (old) {
      if (bridge) {
        old->decSharedBridge_();
      } else {
        old->decSharedBiconnected_();
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
   * Raw pointer.
   */
  template<class U, std::enable_if_t<std::is_base_of_v<U,T>,int> = 0>
  operator U*() {
    return get();
  }

  /**
   * Raw pointer.
   */
  template<class U, std::enable_if_t<std::is_base_of_v<U,T>,int> = 0>
  operator const U*() const {
    return get();
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
    pack(ptr, false);
  }

  /**
   * Set the bridge flag.
   */
  void setBridge() {
    packed.maskOr(BRIDGE);
  }

  /**
   * Unpack raw pointer and flags.
   */
  std::tuple<T*,bool> unpack(const int64_t p) const {
    return std::make_tuple((T*)(p & POINTER), bool(p & BRIDGE));
  }

  /**
   * Unpack raw pointer and flags.
   */
  std::tuple<T*,bool> unpack() const {
    return unpack(packed.load());
  }

  /**
   * Unpack raw pointer and flags, while also setting the lock flag.
   */
  std::tuple<T*,bool> lock() {
    auto p = packed.exchangeOr(LOCK);
    while (p & LOCK) {
      p = packed.exchangeOr(LOCK);
    }
    return unpack(p);
  }

  /**
   * Pack raw pointer and bridge flag. Also releases the lock flag, if set.
   */
  void pack(T* ptr, const bool bridge) {
    packed.store((int64_t(ptr) & POINTER) | (bridge ? BRIDGE : 0));
  }

  /**
   * Packed raw pointer and flags.
   * 
   * @see SharedFlag
   */
  Atomic<int64_t> packed;
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

#include "membirch/Spanner.hpp"
#include "membirch/Bridger.hpp"
#include "membirch/Copier.hpp"
#include "membirch/BiconnectedCopier.hpp"

template<class T>
T* membirch::Shared<T>::get() {
  auto [ptr, bridge] = unpack();
  auto o = ptr;
  if (bridge) {
    std::tie(ptr, bridge) = lock();  // acquire lock and unpack again
    o = ptr;
    if (bridge) {  // if still a bridge, i.e. another thread hasn't copied yet
      if (!o->isUniqueHead_()) {  // last reference optimization
        /* copy biconnected component */
        set_copy();
        o = static_cast<T*>(BiconnectedCopier(o).visitObject(o));
        unset_copy();

        /* replace pointer */
        o->incShared_();
      }
    }
    pack(o, false);  // no longer a bridge, also releases lock
    if (ptr != o) {  // do outside the critical region, as could be expensive
      ptr->decSharedBridge_();
    }
  }
  return o;
}

template<class T>
void membirch::Shared<T>::bridge() {
  Spanner().visit(0, 1, *this);
  Bridger().visit(1, 0, *this);
}

template<class T>
membirch::Shared<T> membirch::Shared<T>::copy() {
  auto [ptr, bridge] = unpack();
  if (!bridge) {
    /* the copy is not necessarily of a biconnected component here, use the
     * general-purpose Copier rather than special-purpose BiconnectedCopier */
    set_copy();
    ptr = static_cast<T*>(Copier().visitObject(ptr));
    unset_copy();
  }
  return Shared<T>(ptr, bridge);
}
