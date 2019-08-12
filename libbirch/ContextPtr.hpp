/**
 * @file
 */
#pragma once

#include "libbirch/SharedPtr.hpp"
#include "libbirch/InitPtr.hpp"
#include "libbirch/Context.hpp"

namespace libbirch {
/**
 * Pointer to a Context object. When the reference is the same as the
 * current context, this acts as a raw pointers, otherwise it acts as a
 * shared pointer.
 *
 * @ingroup libbirch
 */
class ContextPtr {
public:
  /**
   * Default constructor.
   */
  ContextPtr() : pack(0u) {
    //
  }

  /**
   * Value constructor.
   */
  explicit ContextPtr(Context* ptr) {
    set(ptr);
  }

  /**
   * Copy constructor.
   */
  ContextPtr(const ContextPtr& o) {
    set(o.get());
  }

  /**
   * Destructor.
   */
  ~ContextPtr() {
    release();
  }

  /**
   * Copy assignment.
   */
  ContextPtr& operator=(const ContextPtr& o) {
    replace(o.get());
    return *this;
  }

  /**
   * Get the raw pointer.
   */
  Context* get() const {
    /* zero out the cross flag to restore the pointer */
    return reinterpret_cast<Context*>(pack & ~(uintptr_t)1u);
  }

  /**
   * Replace.
   */
  void replace(Context* ptr) {
    auto old = get();
    if (ptr != old) {
      auto oldCross = isCross();
      set(ptr);
      if (old && oldCross) {
        old->decShared();
      }
    }
  }

  /**
   * Release the context.
   */
  void release() {
    if (isCross()) {
      get()->decShared();
    }
    pack = (uintptr_t)0u;
  }

  /**
   * Is this pointer crossed? A crossed pointer is to a context different to
   * that of the context in which it was created (e.g. the context of the
   * object to which it belongs).
   */
  bool isCross() const {
    return pack & (uintptr_t)1u;
  }

  /**
   * Dereference.
   */
  Context& operator*() const {
    return *get();
  }

  /**
   * Member access.
   */
  Context* operator->() const {
    return get();
  }

  /**
   * Equal comparison.
   */
  bool operator==(const ContextPtr& o) const {
    return get() == o.get();
  }

  /**
   * Not equal comparison.
   */
  bool operator!=(const ContextPtr& o) const {
    return get() != o.get();
  }

  /**
   * Is the pointer not null?
   */
  operator bool() const {
    return pack != (uintptr_t)0u;
  }

private:
  /**
   * Set.
   */
  void set(Context* ptr) {
    pack = reinterpret_cast<uintptr_t>(ptr);
    assert(!isCross());
    if (ptr && ptr != currentContext) {
      ptr->incShared();
      pack |= (uintptr_t)1u;
    }
  }

  /**
   * The packed raw pointer to context and cross flag. The least significant
   * bit of the pointer, which would always be zero otherwise (given
   * memory alignment) is used for the cross flag.
   */
  uintptr_t pack;
};
}
