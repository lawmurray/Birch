/**
 * @file
 */
#pragma once

#include "libbirch/SharedPtr.hpp"
#include "libbirch/InitPtr.hpp"
#include "libbirch/Context.hpp"

namespace libbirch {
/**
 * Context-sensitive shared or weak pointer to another Context. It records
 * the Context of the owning object; if this is the same as the referent
 * Context, a weak pointer is kept, otherwise a shared pointer is kept. This
 * is used to avoid reference cycles between Context objects.
 *
 * @ingroup libbirch
 */
class ContextPtr {
public:
  /**
   * Default constructor.
   */
  ContextPtr() :
      owner(currentContext) {
    //
  }

  /**
   * Value constructor.
   */
  ContextPtr(Context* context) :
      context(context == currentContext ? nullptr : context),
      owner(currentContext) {
    //
  }

  /**
   * Copy constructor.
   */
  ContextPtr(const ContextPtr& o) :
      context(nullptr),
      owner(currentContext) {
    if (o.context) {
      if (o.context != owner) {
        context = o.context;
      }
    } else {
      if (o.owner != owner) {
        context = o.owner;
      }
    }
  }

  /**
   * Move constructor.
   */
  ContextPtr(ContextPtr&& o) :
      context(nullptr),
      owner(currentContext) {
    if (o.context) {
      if (o.context != owner) {
        context = std::move(o.context);
      }
    } else {
      if (o.owner != owner) {
        context = std::move(o.owner);
      }
    }
  }

  /**
   * Value assignment.
   */
  ContextPtr& operator=(Context* context) {
    this->context = (context == owner.get()) ? nullptr : context;
    return *this;
  }

  /**
   * Copy assignment.
   */
  ContextPtr& operator=(const ContextPtr& o) {
    if (o.context) {
      if (o.context != owner) {
        context = o.context;
      } else {
        context = nullptr;
      }
    } else {
      if (o.owner != owner) {
        context = o.owner;
      } else {
        context = nullptr;
      }
    }
    return *this;
  }

  /**
   * Move assignment.
   */
  ContextPtr& operator=(ContextPtr&& o) {
    if (o.context) {
      if (o.context != owner) {
        context = std::move(o.context);
      } else {
        context = nullptr;
      }
    } else {
      if (o.owner != owner) {
        context = std::move(o.owner);
      } else {
        context = nullptr;
      }
    }
    return *this;
  }

  /**
   * Get the raw pointer.
   */
  Context* get() const {
    return context.get() ? context.get() : owner.get();
  }

  /**
   * Get the owner.
   */
  Context* getContext() const {
    return owner.get();
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
    return get() != nullptr;
  }

private:
  /**
   * The referent, if it is difference to owner, otherwise `nullptr`.
   */
  SharedPtr<Context> context;

  /**
   * The owner. This is the context in which the pointer itself was created.
   * For a member variable, it is the same as the context of the owning
   * object. For a global or local variable it is the context of the  thread
   * at the time it was created. Use of an InitPtr rather than WeakPtr is
   * sufficient here, as at least a WeakPtr must exist to the owner context
   * elsewhere.
   */
  InitPtr<Context> owner;
};
}
