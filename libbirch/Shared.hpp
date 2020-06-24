/**
 * @file
 */
#pragma once

#include "libbirch/Any.hpp"
#include "libbirch/Atomic.hpp"
#include "libbirch/type.hpp"

namespace libbirch {
/**
 * Shared pointer.
 *
 * @ingroup libbirch
 *
 * @tparam T Type, must derive from Any.
 */
template<class T>
class Shared {
  template<class U> friend class Shared;
  template<class U> friend class Weak;
  template<class U> friend class Init;
  template<class U> friend class Lazy;
public:
  using value_type = T;

  /**
   * Constructor.
   */
  explicit Shared(value_type* ptr = nullptr) :
      packed(pack(ptr, false)) {
    if (ptr) {
      ptr->initShared();
    }
  }

  /**
   * Copy constructor.
   */
  Shared(const Shared& o) {
    T* ptr = std::get<0>(unpack(o.packed.load()));
    if (ptr) {
      ptr->incShared();
    }
    packed.set(pack(ptr, false));
  }

  /**
   * Generic shared pointer copy constructor.
   */
  template<class U, std::enable_if_t<std::is_base_of<T,U>::value,int> = 0>
  Shared(const Shared<U>& o) {
    T* ptr = std::get<0>(unpack(o.packed.load()));
    if (ptr) {
      ptr->incShared();
    }
    packed.set(pack(ptr, false));
  }

  /**
   * Generic other pointer copy constructor.
   */
  template<class Q, std::enable_if_t<std::is_base_of<T,typename Q::value_type>::value,int> = 0>
  Shared(const Q& o) {
    T* ptr = o.get();
    packed.set(pack(ptr, false));
    if (ptr) {
      ptr->incShared();
    }
  }

  /**
   * Move constructor.
   */
  Shared(Shared&& o) {
    T* ptr;
    bool discarded;
    std::tie(ptr, discarded) = unpack(o.packed.maskAnd(int64_t(1)));
    if (ptr && discarded) {
      ptr->restoreShared();
    }
    packed.set(pack(ptr, false));
  }

  /**
   * Generic move constructor.
   */
  template<class U, std::enable_if_t<std::is_base_of<T,U>::value,int> = 0>
  Shared(Shared<U>&& o) {
    T* ptr;
    bool discarded;
    std::tie(ptr, discarded) = unpack(o.packed.maskAnd(int64_t(1)));
    if (ptr && discarded) {
      ptr->restoreShared();
    }
    packed.set(pack(ptr, false));
  }

  /**
   * Destructor.
   */
  ~Shared() {
    release();
  }

  /**
   * Fix after a bitwise copy.
   */
  void bitwiseFix() {
    T* ptr = std::get<0>(unpack(packed.maskAnd(~int64_t(1))));
    if (ptr) {
      ptr->incShared();
    }
  }

  /**
   * Copy assignment.
   */
  Shared& operator=(const Shared& o) {
    assert(!isDiscarded());
    replace(o.get());
    return *this;
  }

  /**
   * Generic shared pointer copy assignment.
   */
  template<class U, std::enable_if_t<std::is_base_of<T,U>::value,int> = 0>
  Shared& operator=(const Shared<U>& o) {
    assert(!isDiscarded());
    replace(o.get());
    return *this;
  }

  /**
   * Generic other pointer copy assignment.
   */
  template<class Q, std::enable_if_t<std::is_base_of<T,typename Q::value_type>::value,int> = 0>
  Shared& operator=(const Q& o) {
    assert(!isDiscarded());
    replace(o.get());
    return *this;
  }

  /**
   * Move assignment.
   */
  Shared& operator=(Shared&& o) {
    assert(!isDiscarded());
    T* ptr;
    bool discarded;
    std::tie(ptr, discarded) = unpack(o.packed.maskAnd(int64_t(1)));
    if (ptr && discarded) {
      ptr->restoreShared();
    }
    auto old = std::get<0>(unpack(packed.exchange(pack(ptr, false))));
    if (old) {
      old->decShared();
    }
    return *this;
  }

  /**
   * Generic move assignment.
   */
  template<class U, std::enable_if_t<std::is_base_of<T,U>::value,int> = 0>
  Shared& operator=(Shared<U>&& o) {
    assert(!isDiscarded());
    T* ptr;
    bool discarded;
    std::tie(ptr, discarded) = unpack(o.packed.maskAnd(int64_t(1)));
    if (ptr && discarded) {
      ptr->restoreShared();
    }
    auto old = std::get<0>(unpack(packed.exchange(pack(ptr, false))));
    if (old) {
      old->decShared();
    }
    return *this;
  }

  /**
   * Is the pointer not null?
   *
   * This is used instead of an `operator bool()` so as not to conflict with
   * conversion operators in the referent type.
   */
  bool query() const {
    return get() != nullptr;
  }

  /**
   * Get the raw pointer.
   */
  T* get() const {
    return std::get<0>(unpack(packed.load()));
  }

  /**
   * Replace.
   */
  void replace(T* ptr) {
    assert(!isDiscarded());
    if (ptr) {
      ptr->initShared();
    }
    auto old = std::get<0>(unpack(packed.exchange(pack(ptr, false))));
    if (old) {
      old->decShared();
    }
  }

  /**
   * Release.
   */
  void release() {
    T* ptr;
    bool discarded;
    std::tie(ptr, discarded) = unpack(packed.maskAnd(int64_t(1)));
    if (ptr) {
      if (discarded) {
        ptr->decMemoShared();
      } else {
        ptr->decShared();
      }
    }
  }
  
  /**
   * Discard.
   */
  void discard() {
    T* ptr;
    bool discarded;
    std::tie(ptr, discarded) = unpack(packed.maskOr(int64_t(1)));
    if (ptr) {
      ptr->discardShared();
    }
  }

  /**
   * Restore.
   */
  void restore() {
    T* ptr;
    bool discarded;
    std::tie(ptr, discarded) = unpack(packed.maskAnd(~int64_t(1)));
    if (ptr) {
      ptr->restoreShared();
    }
  }

  /**
   * Dereference.
   */
  T& operator*() const {
    assert(!isDiscarded());
    return *get();
  }

  /**
   * Member access.
   */
  T* operator->() const {
    assert(!isDiscarded());
    return get();
  }

private:
  /**
   * Combined pointer and discount flag. The rightmost bit is used for the
   * discount flag; this will always be zero for the pointer value given
   * memory alignment.
   */
  Atomic<int64_t> packed;

  /**
   * Is this discarded?
   */
  bool isDiscarded() const {
    return std::get<1>(unpack(packed.load()));
  }

  /**
   * Unpack the pointer and discard flag from an integer.
   */
  static std::pair<T*,bool> unpack(const int64_t packed) {
    T* ptr = (T*)(packed & ~int64_t(1));
    bool discarded = bool(packed & int64_t(1));
    return std::make_pair(ptr, discarded);
  }

  /**
   * Pack the pointer and discard flag into an integer.
   */
  static int64_t pack(const T* ptr, const bool discarded) {
    return int64_t(ptr) | int64_t(discarded);
  }
};

template<class T>
struct is_value<Shared<T>> {
  static const bool value = false;
};

template<class T>
struct is_pointer<Shared<T>> {
  static const bool value = true;
};

template<class T>
struct raw<Shared<T>> {
  using type = T*;
};
}
