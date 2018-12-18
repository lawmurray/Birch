/**
 * @file
 */
#pragma once

#include "libbirch/config.hpp"
#include "libbirch/SharedPtr.hpp"
#include "libbirch/WeakPtr.hpp"
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
      memo(o.get() == top_context() ? nullptr : o.get()),
      context(top_context()) {
    //
  }

  /**
   * Move constructor.
   */
  ContextPtr(ContextPtr&& o) :
      memo(o.get() == top_context() ? nullptr : o.get()),
      context(top_context()) {
    //
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
    this->memo = (o.get() == context.get()) ? nullptr : o.get();
    return *this;
  }

  /**
   * Move assignment.
   */
  ContextPtr& operator=(ContextPtr&& o) {
    this->memo = (o.get() == context.get()) ? nullptr : o.get();
    return *this;
  }

  /**
   * Get the raw pointer.
   */
  Memo* get() const {
    return memo ? memo.get() : context.get();
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
   * The memo. For a member variable of an object created by a clone, this is
   * `nullptr` if it would otherwise equal context, in order to break
   * reference cycles.
   */
  SharedPtr<Memo> memo;

  /**
   * The context. For a member variable of an object created by a clone, this
   * is the context of the object to which it belongs. For anything else,
   * this is `nullptr`. It does not change over the lifetime of this.
   */
  WeakPtr<Memo> context;
};
}
