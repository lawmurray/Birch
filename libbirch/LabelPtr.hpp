/**
 * @file
 */
#pragma once

#include "libbirch/Label.hpp"
#include "libbirch/Atomic.hpp"

namespace libbirch {
/**
 * Pointer to a Label.
 *
 * @ingroup libbirch
 */
class LabelPtr {
public:
  /**
   * Constructor.
   */
  explicit LabelPtr(Label* ptr = nullptr) :
      packed(pack(ptr, false)) {
    if (ptr) {
      ptr->incUsage();
    }
  }

  /**
   * Copy constructor.
   */
  LabelPtr(const LabelPtr& o) {
    auto ptr = std::get<0>(unpack(o.packed.load()));
    if (ptr) {
      ptr->incUsage();
    }
    packed.set(pack(ptr, false));
  }

  /**
   * Move constructor.
   */
  LabelPtr(LabelPtr&& o) {
    Label* ptr;
    bool discarded;
    std::tie(ptr, discarded) = unpack(o.packed.maskAnd(int64_t(1)));
    if (ptr && discarded) {
      ptr->incUsage();
    }
    packed.set(pack(ptr, false));
  }

  /**
   * Destructor.
   */
  ~LabelPtr() {
    release();
  }

  /**
   * Fix after a bitwise copy.
   */
  void bitwiseFix(Label* ptr) {
    if (ptr) {
      ptr->incUsage();
    }
    packed.set(pack(ptr, false));
  }

  /**
   * Copy assignment.
   */
  LabelPtr& operator=(const LabelPtr& o) {
    assert(!isDiscarded());
    replace(o.get());
    return *this;
  }

  /**
   * Move assignment.
   */
  LabelPtr& operator=(LabelPtr&& o) {
    assert(!isDiscarded());
    Label* ptr;
    bool discarded;
    std::tie(ptr, discarded) = unpack(o.packed.maskAnd(int64_t(1)));
    if (ptr && discarded) {
      ptr->incUsage();
    }
    auto old = std::get<0>(unpack(packed.exchange(pack(ptr, false))));
    if (old && old->decUsage() == 0u) {
      delete old;
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
  Label* get() const {
    return std::get<0>(unpack(packed.load()));
  }

  /**
   * Replace.
   */
  void replace(Label* ptr) {
    assert(!isDiscarded());
    if (ptr) {
      ptr->incUsage();
    }
    auto old = std::get<0>(unpack(packed.exchange(pack(ptr, false))));
    if (old && old->decUsage() == 0u) {
      delete old;
    }
  }

  /**
   * Release.
   */
  void release() {
    Label* ptr;
    bool discarded;
    std::tie(ptr, discarded) = unpack(packed.maskAnd(int64_t(1)));
    if (ptr && !discarded && ptr->decUsage() == 0u) {
      delete ptr;
    }
  }

  /**
   * Discard.
   */
  void discard() {
    Label* ptr;
    bool discarded;
    std::tie(ptr, discarded) = unpack(packed.maskOr(int64_t(1)));
    if (ptr && !discarded && ptr->decUsage() == 0u) {
      delete ptr;
    }
  }

  /**
   * Restore.
   */
  void restore() {
    Label* ptr;
    bool discarded;
    std::tie(ptr, discarded) = unpack(packed.maskAnd(~int64_t(1)));
    if (ptr && discarded) {
      ptr->incUsage();
    }
  }

  /**
   * Dereference.
   */
  Label& operator*() const {
    assert(!isDiscarded());
    return *get();
  }

  /**
   * Member access.
   */
  Label* operator->() const {
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
  static std::pair<Label*,bool> unpack(const int64_t packed) {
    Label* ptr = (Label*)(packed & ~int64_t(1));
    bool discarded = bool(packed & int64_t(1));
    return std::make_pair(ptr, discarded);
  }

  /**
   * Pack the pointer and discard flag into an integer.
   */
  static int64_t pack(const Label* ptr, const bool discarded) {
    return int64_t(ptr) | int64_t(discarded);
  }
};
}
