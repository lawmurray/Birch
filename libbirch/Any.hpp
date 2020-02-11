/**
 * @file
 */
#pragma once

#include "libbirch/external.hpp"
#include "libbirch/class.hpp"
#include "libbirch/Counted.hpp"
#include "libbirch/Atomic.hpp"

namespace libbirch {
class Label;

/**
 * Base for all class types when lazy deep clone is used.
 *
 * @ingroup libbirch
 */
class Any: public Counted {
public:
  using class_type_ = Any;
  using this_type_ = Any;

  Any(const Any&) = delete;
  Any& operator=(const Any&) = delete;

  /**
   * Constructor.
   *
   * @param context Current context.
   */
  Any(Label* context);

  /**
   * Deep copy constructor.
   *
   * @param context Current context.
   * @param label Label associated with clone.
   * @param o Source object.
   */
  Any(Label* context, Label* label, const Any& o);

  /**
   * Destructor.
   */
  virtual ~Any();

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
  Label* getLabel() const;

  /**
   * Deep freeze.
   */
  void freeze();

  /**
   * Shallow thaw to allow reuse of the object.
   *
   * @param label The new label of the object.
   */
  void thaw(Label* label);

  /**
   * Deep finish of lazy clone.
   */
  void finish();

  virtual Any* clone_(Label* context) const = 0;

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
  virtual void doThaw_(Label* label);

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

namespace bi {
  namespace type {
template<>
struct super_type<libbirch::Any> {
  using type = libbirch::Counted;
};
  }
}

inline libbirch::Any::Any(Label* context) :
    Counted(),
    label((intptr_t)context),
    frozen(false),
    finished(false)
    #if ENABLE_SINGLE_REFERENCE_OPTIMIZATION
    , single(false)
    #endif
    {
  assert(context);
}

inline libbirch::Any::Any(Label* context, Label* label,
    const Any& o) :
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

inline libbirch::Any::~Any() {
  //
}

inline bool libbirch::Any::isFrozen() const {
  return frozen;
}

inline bool libbirch::Any::isFinished() const {
  return finished;
}

inline bool libbirch::Any::isSingle() const {
  #if ENABLE_SINGLE_REFERENCE_OPTIMIZATION
  return single;
  #else
  return false;
  #endif
}

inline libbirch::Label* libbirch::Any::getLabel() const {
  return (Label*)label;
}

inline void libbirch::Any::freeze() {
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

inline void libbirch::Any::thaw(Label* label) {
  this->label = (intptr_t)label;
  frozen = false;
  finished = false;
  #if ENABLE_SINGLE_REFERENCE_OPTIMIZATION
  single = false;
  #endif
  doThaw_(label);
}

inline void libbirch::Any::finish() {
  if (!finished) {
    finished = true;
    if (numShared() > 0u) {
      doFinish_();
    }
  }
}

inline void libbirch::Any::doFreeze_() {
  //
}

inline void libbirch::Any::doThaw_(Label* label) {
  //
}

inline void libbirch::Any::doFinish_() {
  //
}
