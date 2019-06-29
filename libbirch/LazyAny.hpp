/**
 * @file
 */
#if ENABLE_LAZY_DEEP_CLONE
#pragma once

#include "libbirch/external.hpp"
#include "libbirch/Counted.hpp"
#include "libbirch/WeakPtr.hpp"
#include "libbirch/Atomic.hpp"
#include "libbirch/ExclusiveLock.hpp"

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
   * Get the context in which this object was created.
   */
  LazyContext* getContext();

  /**
   * If the object is frozen, return the object to which to forward writes.
   */
  LazyAny* getForward();

  /**
   * If the object is frozen, and the object to which to forward writes has
   * already been created, return it, otherwise return `this`. This is used
   * to forward reads, ensuring that the most recent writes are visible.
   */
  LazyAny* pullForward();

  /**
   * Finish any remaining lazy deep clones in the subgraph reachable from the
   * object.
   */
  void finish();

  /**
   * Name of the class.
   */
  virtual const char* name_() const {
    return "Any";
  }

protected:
  /**
   * Perform the actual finish of the object. This is overwritten by derived
   * classes.
   */
  virtual void doFinish_();

  /**
   * Context in which this object was created.
   */
  WeakPtr<LazyContext> context;

  /**
   * If frozen, object to which to forward. This must be thread safe, and
   * so an atomic raw pointer is used, with manual shared reference count
   * maintenance.
   */
  SharedPtr<LazyAny> forward;

  /**
   * Have clones of all objects reachable from this object finished?
   */
  Atomic<bool> finished;

  /**
   * Lock.
   */
  ExclusiveLock mutex;
};
}

inline libbirch::LazyAny::LazyAny() :
    Counted(),
    context(currentContext),
    forward(nullptr),
    finished(false) {
  //
}

inline libbirch::LazyAny::LazyAny(const LazyAny& o) :
    Counted(o),
    context(currentContext),
    forward(nullptr),
    finished(false) {
  //
}

inline libbirch::LazyAny::~LazyAny() {
  //
}

inline libbirch::LazyContext* libbirch::LazyAny::getContext() {
  return context.get();
}

inline void libbirch::LazyAny::finish() {
  if (!finished.exchange(true) && sharedCount.load() > 0u) {
    doFinish_();
  }
}

inline void libbirch::LazyAny::doFinish_() {
  //
}

#endif
