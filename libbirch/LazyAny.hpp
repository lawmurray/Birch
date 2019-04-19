/**
 * @file
 */
#if ENABLE_LAZY_DEEP_CLONE
#pragma once

#include "libbirch/Counted.hpp"
#include "libbirch/WeakPtr.hpp"

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
  LazyMemo* getContext();

  /**
   * If this memo is frozen, return the memo to which it should forward,
   * otherwise `this`.
   */
  LazyAny* getForward();

  /**
   * If this memo is frozen, and the memo to which it should forward has
   * already been created, return that memoo, otherwise `this`.
   */
  LazyAny* pullForward();

  /**
   * Finish any remaining lazy deep clones in the member variables of this
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
   * Memo responsible for the creation of this object.
   */
  WeakPtr<LazyMemo> context;

  /**
   * If frozen, object to which to forward. This must be thread safe, and
   * so an atomic raw pointer is used, with manual shared reference count
   * maintenance.
   */
  std::atomic<LazyAny*> forward;

  /**
   * Have clones of all objects reachable from this object finished?
   */
  std::atomic<bool> finished;
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
  auto forward1 = this->forward.load(std::memory_order_relaxed);
  if (forward1) {
    forward1->decShared();
  }
}

inline libbirch::LazyMemo* libbirch::LazyAny::getContext() {
  return context.get();
}

inline void libbirch::LazyAny::finish() {
  bool expected = false;
  bool desired = true;
  if (finished.compare_exchange_strong(expected, desired,
        std::memory_order_relaxed)) {
    if (sharedCount > 0) {
      doFinish_();
    }
  }
}

inline void libbirch::LazyAny::doFinish_() {
  //
}

#endif
