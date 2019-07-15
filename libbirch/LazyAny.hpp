/**
 * @file
 */
#if ENABLE_LAZY_DEEP_CLONE
#pragma once

#include "libbirch/external.hpp"
#include "libbirch/Counted.hpp"
#include "libbirch/WeakPtr.hpp"
#include "libbirch/Atomic.hpp"

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
   * Is the object frozen? This returns true if either a freeze is in
   * progress (i.e. another thread is in the process of freezing the object),
   * or if the freeze is complete.
   */
  bool isFrozen() const;

  /**
   * Is the object finished?
   */
  bool isFinished() const;

  /**
   * If frozen, at the time of freezing, was the reference count only one?
   */
  bool isSingle() const;

  /**
   * Has this object been added to the right hand side of a memo?
   */
  bool isMemo() const;

  /**
   * Get the context in which this object was created.
   */
  LazyContext* getContext();

  /**
   * Freeze.
   */
  void freeze();

  /**
   * Finish any remaining lazy deep clones in the subgraph reachable from
   * this.
   */
  void finish();

  /**
   * Indicate that the single-reference optimization can no longer be applied
   * to this object, as the reference count within a single context has been
   * increased.
   */
  void multiply();

  /**
   * Indicate that this object has been added to the right hand side of a
   * memo.
   */
  void memoize();

  /**
   * Name of the class.
   */
  virtual const char* name_() const {
    return "Any";
  }

protected:
  /**
   * Perform the actual freeze of the object. This is overwritten by derived
   * classes. The non-virtual freeze() handles thread safety so that this
   * need not.
   */
  virtual void doFreeze_();

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
   * Is this frozen (read-only)?
   */
  Atomic<bool> frozen;

  /**
   * Is this finished?
   */
  Atomic<bool> finished;

  #if ENABLE_SINGLE_REFERENCE_OPTIMIZATION
  /**
   * If frozen, at the time of freezing, was the reference count only one?
   */
  Atomic<bool> single;

  /**
   * Has this object been added to the right hand side of a memo?
   */
  bool memo;
  #endif
};
}

inline libbirch::LazyAny::LazyAny() :
    Counted(),
    context(currentContext),
    frozen(false),
    finished(false)
    #if ENABLE_SINGLE_REFERENCE_OPTIMIZATION
    , single(false), memo(false)
    #endif
    {
  //
}

inline libbirch::LazyAny::LazyAny(const LazyAny& o) :
    Counted(o),
    context(currentContext),
    frozen(false),
    finished(false)
    #if ENABLE_SINGLE_REFERENCE_OPTIMIZATION
    , single(false), memo(false)
    #endif
    {
  //
}

inline libbirch::LazyAny::~LazyAny() {
  //
}

inline bool libbirch::LazyAny::isFrozen() const {
  return frozen.load();
}

inline bool libbirch::LazyAny::isFinished() const {
  return finished.load();
}

inline bool libbirch::LazyAny::isSingle() const {
  #if ENABLE_SINGLE_REFERENCE_OPTIMIZATION
  return single.load();
  #else
  return false;
  #endif
}

inline bool libbirch::LazyAny::isMemo() const {
  #if ENABLE_SINGLE_REFERENCE_OPTIMIZATION
  return memo;
  #else
  return false;
  #endif
}

inline libbirch::LazyContext* libbirch::LazyAny::getContext() {
  return context.get();
}

inline void libbirch::LazyAny::freeze() {
  if (!frozen.exchange(true)) {
    auto nshared = numShared();
    #if ENABLE_SINGLE_REFERENCE_OPTIMIZATION
    single.store(!isMemo() && nshared <= 1u && numWeak() <= 1u);
    #endif
    if (nshared > 0u) {
      doFreeze_();
    }
  }
}

inline void libbirch::LazyAny::finish() {
  if (!finished.exchange(true)) {
    if (numShared() > 0u) {
      doFinish_();
    }
  }
}

inline void libbirch::LazyAny::multiply() {
  #if ENABLE_SINGLE_REFERENCE_OPTIMIZATION
  single.store(false);
  #endif
}

inline void libbirch::LazyAny::memoize() {
  #if ENABLE_SINGLE_REFERENCE_OPTIMIZATION
  memo = true;
  #endif
}

inline void libbirch::LazyAny::doFreeze_() {
  //
}

inline void libbirch::LazyAny::doFinish_() {
  //
}

#endif
