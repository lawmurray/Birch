/**
 * @file
 */
#if ENABLE_LAZY_DEEP_CLONE
#pragma once

#include "libbirch/external.hpp"
#include "libbirch/Counted.hpp"
#include "libbirch/InitPtr.hpp"
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
   * Deep copy constructor.
   */
  LazyAny(const LazyAny& o, int);

  /**
   * Destructor.
   */
  virtual ~LazyAny();

  LazyAny(const LazyAny&) = delete;
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
   * Get the context in which this object was created.
   */
  LazyContext* getContext();

  /**
   * Deep freeze.
   */
  void freeze();

  /**
   * Shallow thaw to allow reuse by another context.
   *
   * @param context The new context of the object.
   */
  void thaw(LazyContext* context);

  /**
   * Deep finish of lazy clone.
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
   * Perform the actual freeze of the object. This is overwritten by derived
   * classes.
   */
  virtual void doFreeze_();

  /**
   * Perform the actual thaw of the object. This is overwritten by derived
   * classes.
   */
  virtual void doThaw_(LazyContext* context);

  /**
   * Perform the actual finish of the object. This is overwritten by derived
   * classes.
   */
  virtual void doFinish_();

  /**
   * Context in which this object was created.
   */
  intptr_t context:61;

  /**
   * Is this frozen (read-only)?
   */
  bool frozen:1;

  /**
   * Is this finished?
   */
  bool finished:1;

  #if ENABLE_SINGLE_REFERENCE_OPTIMIZATION
  /**
   * If frozen, at the time of freezing, was the reference count only one?
   */
  bool single:1;
  #endif
};
}

inline libbirch::LazyAny::LazyAny() :
    Counted(),
    context((intptr_t)currentContext),
    frozen(false),
    finished(false)
    #if ENABLE_SINGLE_REFERENCE_OPTIMIZATION
    , single(false)
    #endif
    {
  //
}

inline libbirch::LazyAny::LazyAny(const LazyAny& o, int) :
    Counted(o, 0),
    context((intptr_t)currentContext),
    frozen(false),
    finished(false)
    #if ENABLE_SINGLE_REFERENCE_OPTIMIZATION
    , single(false)
    #endif
    {
  //
}

inline libbirch::LazyAny::~LazyAny() {
  //
}

inline bool libbirch::LazyAny::isFrozen() const {
  return frozen;
}

inline bool libbirch::LazyAny::isFinished() const {
  return finished;
}

inline bool libbirch::LazyAny::isSingle() const {
  #if ENABLE_SINGLE_REFERENCE_OPTIMIZATION
  return single;
  #else
  return false;
  #endif
}

inline libbirch::LazyContext* libbirch::LazyAny::getContext() {
  return (LazyContext*)context;
}

inline void libbirch::LazyAny::freeze() {
  if (!frozen) {
    frozen = true;
    auto nshared = numShared();
    #if ENABLE_SINGLE_REFERENCE_OPTIMIZATION
    single = nshared <= 1u && numWeak() <= 1u;
    #endif
    if (nshared > 0u) {
      doFreeze_();
    }
  }
}

inline void libbirch::LazyAny::thaw(LazyContext* context) {
  this->context = (intptr_t)context;
  frozen = false;
  finished = false;
  #if ENABLE_SINGLE_REFERENCE_OPTIMIZATION
  single = false;
  #endif
  doThaw_(context);
}

inline void libbirch::LazyAny::finish() {
  if (!finished) {
    finished = true;
    if (numShared() > 0u) {
      doFinish_();
    }
  }
}

inline void libbirch::LazyAny::doFreeze_() {
  //
}

inline void libbirch::LazyAny::doThaw_(LazyContext* context) {
  //
}

inline void libbirch::LazyAny::doFinish_() {
  //
}

#endif
