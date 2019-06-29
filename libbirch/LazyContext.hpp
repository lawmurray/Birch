/**
 * @file
 */
#pragma once
#if ENABLE_LAZY_DEEP_CLONE

#include "libbirch/Counted.hpp"
#include "libbirch/LazyAny.hpp"
#include "libbirch/SharedPtr.hpp"
#include "libbirch/Map.hpp"
#include "libbirch/ReadWriteLock.hpp"

namespace libbirch {
/**
 * Context for lazy deep clones.
 *
 * @ingroup libbirch
 */
class LazyContext: public Counted {
  friend class List;
public:
  using class_type_ = LazyContext;

protected:
  /**
   * Constructor for root node.
   */
  LazyContext();

  /**
   * Constructor for non-root node.
   *
   * @param parent Parent.
   */
  LazyContext(LazyContext* parent);

  /**
   * Destructor.
   */
  virtual ~LazyContext();

public:
  libbirch_create_function_
  libbirch_emplace_function_
  libbirch_destroy_function_

  /**
   * Fork to create a new child context.
   *
   * @return The child context.
   */
  LazyContext* fork();

  /**
   * Map an object that may not yet have been cloned, cloning it if
   * necessary.
   */
  LazyAny* get(LazyAny* o);

  /**
   * Finish the cloning of an object.
   */
  LazyAny* finish(LazyAny* o);

  /**
   * Map an object that may not yet have been cloned, without cloning it.
   * This is used as an optimization for read-only access.
   */
  LazyAny* pull(LazyAny* o);

  /**
   * Shallow copy.
   */
  LazyAny* copy(LazyAny* o);

protected:
  virtual void doFreeze_();

private:
  /**
   * Memo that maps source objects to clones.
   */
  Map m;

  /**
   * Lock.
   */
  ReadWriteLock l;
};
}

inline libbirch::LazyContext* libbirch::LazyContext::fork() {
  return create_(this);
}

inline void libbirch::LazyContext::doFreeze_() {
  l.read();
  m.freeze();
  l.unread();
}

#endif
