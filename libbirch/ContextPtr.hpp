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
  ContextPtr() : ptr(0), cross(false) {
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
    return reinterpret_cast<Context*>(ptr);
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
    ptr = 0;
    cross = false;
  }

  /**
   * Is this pointer crossed? A crossed pointer is to a context different to
   * that of the context in which it was created (e.g. the context of the
   * object to which it belongs).
   */
  bool isCross() const {
    return cross;
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
    return ptr != 0;
  }

private:
  /**
   * Set.
   */
  void set(Context* ptr) {
    this->ptr = reinterpret_cast<intptr_t>(ptr);
    cross = ptr && ptr != currentContext;
    if (cross) {
      ptr->incShared();
    }
  }

  /**
   * Raw pointer.
   */
  intptr_t ptr:63;

  /**
   * Is this a cross pointer?
   */
  bool cross:1;
};
}
