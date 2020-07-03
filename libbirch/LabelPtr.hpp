/**
 * @file
 */
#pragma once

#include "libbirch/Atomic.hpp"
#include "libbirch/memory.hpp"
#include "libbirch/type.hpp"

namespace libbirch {
class Label;

/**
 * LabelPtr pointer to a Label object. Provides some optimizations over
 * LabelPtr<Label>, e.g. reference counts to the root label need not be
 * updated.
 *
 * @ingroup libbirch
 */
class LabelPtr {
public:
  using value_type = Label;

  /**
   * Constructor.
   */
  explicit LabelPtr(Label* ptr = nullptr);

  /**
   * Copy constructor.
   */
  LabelPtr(const LabelPtr& o);

  /**
   * Move constructor.
   */
  LabelPtr(LabelPtr&& o);

  /**
   * Destructor.
   */
  ~LabelPtr();

  /**
   * Fix after a bitwise copy.
   */
  void bitwiseFix();

  /**
   * Copy assignment.
   */
  LabelPtr& operator=(const LabelPtr& o);

  /**
   * Move assignment.
   */
  LabelPtr& operator=(LabelPtr&& o);

  /**
   * Is the pointer not null?
   *
   * This is used instead of an `operator bool()` so as not to conflict with
   * conversion operators in the referent type.
   */
  bool query() const;

  /**
   * Get the raw pointer.
   */
  Label* get() const;

  /**
   * Replace.
   */
  void replace(Label* ptr);

  /**
   * Release.
   */
  void release();

  /**
   * Dereference.
   */
  Label& operator*() const;

  /**
   * Member access.
   */
  Label* operator->() const;

  /**
   * Mark.
   */
  void mark();

  /**
   * Scan.
   */
  void scan();

  /**
   * Reach.
   */
  void reach();

  /**
   * Collect.
   */
  void collect();

private:
  /**
   * Raw pointer.
   */
  Atomic<Label*> ptr;
};

template<>
struct is_value<LabelPtr> {
  static const bool value = false;
};

template<>
struct is_pointer<LabelPtr> {
  static const bool value = true;
};

template<>
struct raw<LabelPtr> {
  using type = Label*;
};
}
