/**
 * @file
 */
#pragma once

#include "libbirch/external.hpp"
#include "libbirch/class.hpp"
#include "libbirch/Counted.hpp"
#include "libbirch/Cloner.hpp"

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

  /**
   * Constructor.
   */
  Any();

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
   * Clone function.
   */
  virtual Any* clone_() const = 0;

  /**
   * Accept function.
   */
  template<class V>
  void accept_(V& v) {
    //
  }

protected:
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

inline libbirch::Any::Any() :
    Counted(),
    label((intptr_t)0),
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
