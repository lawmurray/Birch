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
      ptr->incShared();
    }
  }

  /**
   * Copy constructor.
   */
  Shared(const Shared& o) {
    T* ptr;
    std::tie(ptr, std::ignore) = unpack(o.packed.load());
    packed.set(pack(ptr, false));
    if (ptr) {
      ptr->incShared();
    }
  }

  /**
   * Generic shared pointer copy constructor.
   */
  template<class U, std::enable_if_t<std::is_base_of<T,U>::value,int> = 0>
  Shared(const Shared<U>& o) {
    T* ptr;
    std::tie(ptr, std::ignore) = unpack(o.packed.load());
    packed.set(pack(ptr, false));
    if (ptr) {
      ptr->incShared();
    }
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
    T* ptr;
    bool discarded;
    std::tie(ptr, discarded) = unpack(packed.get());
    // ^ get() not load(), as no need to be thread safe here
    if (ptr) {
      if (discarded) {
        ptr->incMemoShared();
      } else {
        ptr->incShared();
      }
    }
  }

  /**
   * Copy assignment.
   */
  Shared& operator=(const Shared& o) {
    replace(o.get());
    return *this;
  }

  /**
   * Generic shared pointer copy assignment.
   */
  template<class U, std::enable_if_t<std::is_base_of<T,U>::value,int> = 0>
  Shared& operator=(const Shared<U>& o) {
    replace(o.get());
    return *this;
  }

  /**
   * Generic other pointer copy assignment.
   */
  template<class Q, std::enable_if_t<std::is_base_of<T,typename Q::value_type>::value,int> = 0>
  Shared& operator=(const Q& o) {
    replace(o.get());
    return *this;
  }

  /**
   * Move assignment.
   */
  Shared& operator=(Shared&& o) {
    /* assume that the pointer is not discarded at first; once the atomic
     * write is complete, if it is meant to be discarded, then discard */
    T* ptr;
    T* old;
    bool discarded;
    std::tie(ptr, discarded) = unpack(o.packed.maskAnd(int64_t(1)));
    if (ptr && discarded) {
      ptr->restoreShared();
    }
    std::tie(old, discarded) = unpack(packed.exchange(pack(ptr, false)));
    if (old) {
      if (discarded) {
        old->decMemoShared();
      } else {
        old->decShared();
      }
    }
    if (discarded) {
      discard();  // was meant to be discarded
    }
    return *this;
  }

  /**
   * Generic move assignment.
   */
  template<class U, std::enable_if_t<std::is_base_of<T,U>::value,int> = 0>
  Shared& operator=(Shared<U>&& o) {
    /* assume that the pointer is not discarded at first; once the atomic
     * write is complete, if it is meant to be discarded, then discard */
    T* ptr;
    T* old;
    bool discarded;
    std::tie(ptr, discarded) = unpack(o.packed.maskAnd(int64_t(1)));
    if (ptr && discarded) {
      ptr->restoreShared();
    }
    std::tie(old, discarded) = unpack(packed.exchange(pack(ptr, false)));
    if (old) {
      if (discarded) {
        old->decMemoShared();
      } else {
        old->decShared();
      }
    }
    if (discarded) {
      discard();  // was meant to be discarded
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
    T* ptr;
    std::tie(ptr, std::ignore) = unpack(packed.load());
    return ptr;
  }

  /**
   * Replace.
   */
  void replace(T* ptr) {
    /* assume that the pointer is not discarded at first; once the atomic
     * write is complete, if it is meant to be discarded, then discard */
    if (ptr) {
      ptr->incShared();
    }
    T* old;
    bool discarded;
    std::tie(old, discarded) = unpack(packed.exchange(pack(ptr, false)));
    if (old) {
      if (discarded) {
        old->decMemoShared();
      } else {
        old->decShared();
      }
    }
    if (discarded) {
      discard();  // was meant to be discarded
    }
  }

  /**
   * Release.
   */
  void release() {
    T* old;
    bool discarded;
    std::tie(old, discarded) = unpack(packed.maskAnd(int64_t(1)));
    if (old) {
      if (discarded) {
        old->decMemoShared();
      } else {
        old->decShared();
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
    if (ptr && !discarded) {
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
    if (ptr && discarded) {
      ptr->restoreShared();
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

private:
  /**
   * Combined pointer and discount flag. The rightmost bit is used for the
   * discount flag; this will always be zero for the pointer value given
   * memory alignment.
   */
  Atomic<int64_t> packed;

  /**
   * Unpack the pointer and discard flag from storage.
   */
  static std::pair<T*,bool> unpack(const int64_t packed) {
    T* ptr = (T*)(packed & ~int64_t(1));
    bool discarded = bool(packed & int64_t(1));
    return std::make_pair(ptr, discarded);
  }

  /**
   * Pack the pointer and discard flag into storage.
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
