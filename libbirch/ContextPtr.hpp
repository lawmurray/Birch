/**
 * @file
 */
#pragma once

#include "libbirch/config.hpp"
#include "libbirch/SharedPtr.hpp"
#include "libbirch/InitPtr.hpp"
#include "libbirch/Memo.hpp"

namespace bi {
/**
 * Shared or weak pointer to a Memo, according to context. This is used by
 * SharedCOW and WeakCOW for the memo field. It records the context of the
 * pointer: whether it is part of a member variable or not, and keeps a
 * weak or shared pointer to the Memo according to that context.
 *
 * @ingroup libbirch
 */
class ContextPtr {
public:
  /**
   * Value constructor.
   */
  ContextPtr(Memo* memo = nullptr) :
      memo(memo == top_context() ? nullptr : memo),
      context(top_context()) {
    //
  }

  /**
   * Copy constructor.
   */
  ContextPtr(const ContextPtr& o) :
      memo(nullptr),
      context(top_context()) {
    if (o.memo) {
      if (o.memo != context) {
        memo = o.memo;
      }
    } else if (o.context != context) {
      memo = o.context;
    }
  }

  /**
   * Move constructor.
   */
  ContextPtr(ContextPtr&& o) :
      memo(nullptr),
      context(top_context()) {
    if (o.memo) {
      if (o.memo != context) {
        memo = std::move(o.memo);
      }
    } else if (o.context != context) {
      memo = std::move(o.context);
    }
  }

  /**
   * Value assignment.
   */
  ContextPtr& operator=(Memo* memo) {
    this->memo = (memo == context.get()) ? nullptr : memo;
    return *this;
  }

  /**
   * Copy assignment.
   */
  ContextPtr& operator=(const ContextPtr& o) {
    if (o.memo) {
      if (o.memo != context) {
        memo = o.memo;
      }
    } else if (o.context != context) {
      memo = o.context;
    } else {
      memo = nullptr;
    }
    return *this;
  }

  /**
   * Move assignment.
   */
  ContextPtr& operator=(ContextPtr&& o) {
    if (o.memo) {
      if (o.memo != context) {
        memo = std::move(o.memo);
      }
    } else if (o.context != context) {
      memo = std::move(o.context);
    } else {
      memo = nullptr;
    }
    return *this;
  }

  /**
   * Get the raw pointer.
   */
  Memo* get() const {
    return memo.get() ? memo.get() : context.get();
  }

  /**
   * Dereference.
   */
  Memo& operator*() const {
    return *get();
  }

  /**
   * Member access.
   */
  Memo* operator->() const {
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
   * The memo, if it is difference to context, otherwise `nullptr`.
   */
  SharedPtr<Memo> memo;

  /**
   * The owning context. This is the context in which the pointer itself was
   * created. For a member variable, it is the same as the context of the
   * containing object.
   *
   * It is sufficient to use InitPtr instead of WeakPtr, as a WeakPtr to the
   * context should exist elsewhere.
   */
  InitPtr<Memo> context;
};
}
