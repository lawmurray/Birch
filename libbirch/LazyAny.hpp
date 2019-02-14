/**
 * @file
 */
#if ENABLE_LAZY_DEEP_CLONE
#pragma once

#include "libbirch/Counted.hpp"
#include "libbirch/InitPtr.hpp"

#include <atomic>

namespace bi {
/**
 * Base for all class types when lazy deep clone is used.
 *
 * @ingroup libbirch
 */
class LazyAny: public Counted {
public:
  using class_type = LazyAny;
  using this_type = LazyAny;

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
  STANDARD_CREATE_FUNCTION
  STANDARD_EMPLACE_FUNCTION
  STANDARD_CLONE_FUNCTION
  STANDARD_DESTROY_FUNCTION

  /**
   * If this object is frozen, return the object to which it should forward,
   * otherwise `this`.
   */
  LazyAny* getForward();

  /**
   * If this object is frozen, and the object to which it should forward has
   * already been created, return that object, otherwise `this`.
   */
  LazyAny* pullForward();

  /**
   * Get the memo responsible for the creation of this object.
   */
  Memo* getContext();

protected:
  /**
   * Memo responsible for the creation of this object.
   */
  InitPtr<Memo> context;

  /**
   * If frozen, object to which to forward. This must be thread safe, and
   * so an atomic raw pointer is used, with manual shared reference count
   * maintenance.
   */
  std::atomic<LazyAny*> forward;
};
}

inline bi::Memo* bi::LazyAny::getContext() {
  return context.get();
}

#endif
