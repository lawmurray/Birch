/**
 * @file
 */
#if ENABLE_LAZY_DEEP_CLONE
#pragma once

#include "libbirch/Counted.hpp"
#include "libbirch/InitPtr.hpp"

#include <atomic>

namespace libbirch {
/**
 * Base for all class types when lazy deep clone is used.
 *
 * @ingroup libbirch
 */
class LazyAny: public Counted {
public:
  using class_type_ = LazyAny;
  using this_type_ = LazyAny;

protected:
  /**
   * Constructor.
   */
  LazyAny();

  /**
   * Copy constructor.
   */
  LazyAny(const LazyAny& o);

  /**
   * Destructor.
   */
  virtual ~LazyAny();

  /**
   * Copy assignment operator.
   */
  LazyAny& operator=(const LazyAny&) = delete;

public:
  libbirch_create_function_
  libbirch_emplace_function_
  libbirch_clone_function_
  libbirch_destroy_function_

  /**
   * Get the memo responsible for the creation of this object.
   */
  Memo* getContext();

protected:
  /**
   * Memo responsible for the creation of this object.
   */
  InitPtr<Memo> context;
};
}

inline libbirch::Memo* libbirch::LazyAny::getContext() {
  return context.get();
}

#endif
