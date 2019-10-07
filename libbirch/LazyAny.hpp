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
class LazyLabel;

/**
 * Base for all class types when lazy deep clone is used.
 *
 * @ingroup libbirch
 */
class LazyAny: public Counted {
public:
  using class_type_ = LazyAny;
  using this_type_ = LazyAny;

  LazyAny(const LazyAny&) = delete;
  LazyAny& operator=(const LazyAny&) = delete;

  /**
   * Constructor.
   */
  LazyAny(LazyLabel* label);

  /**
   * Deep copy constructor.
   */
  LazyAny(LazyLabel* label, const LazyAny& o);

  /**
   * Destructor.
   */
  virtual ~LazyAny();

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
   * Get the label assigned to the object.
   */
  LazyLabel* getLabel();

  /**
   * Deep freeze.
   */
  void freeze();

  /**
   * Shallow thaw to allow reuse of the object.
   *
   * @param label The new label of the object.
   */
  void thaw(LazyLabel* label);

  /**
   * Deep finish of lazy clone.
   */
  void finish();

  virtual LazyAny* clone_(LazyLabel* label) const {
    return new LazyAny(label, *this);
  }

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
  virtual void doThaw_(LazyLabel* label);

  /**
   * Perform the actual finish of the object. This is overwritten by derived
   * classes.
   */
  virtual void doFinish_();

  /**
   * Label of the object.
   */
  intptr_t label:61;

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

inline libbirch::LazyAny::LazyAny(LazyLabel* label) :
    Counted(),
    label((intptr_t)label),
    frozen(false),
    finished(false)
    #if ENABLE_SINGLE_REFERENCE_OPTIMIZATION
    , single(false)
    #endif
    {
  //
}

inline libbirch::LazyAny::LazyAny(LazyLabel* label, const LazyAny& o) :
    Counted(o),
    label((intptr_t)label),
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

inline libbirch::LazyLabel* libbirch::LazyAny::getLabel() {
  return (LazyLabel*)label;
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

inline void libbirch::LazyAny::thaw(LazyLabel* label) {
  this->label = (intptr_t)label;
  frozen = false;
  finished = false;
  #if ENABLE_SINGLE_REFERENCE_OPTIMIZATION
  single = false;
  #endif
  doThaw_(label);
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

inline void libbirch::LazyAny::doThaw_(LazyLabel* label) {
  //
}

inline void libbirch::LazyAny::doFinish_() {
  //
}

#endif
