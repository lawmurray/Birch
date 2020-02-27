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

  /**
   * If frozen, at the time of freezing, was the reference count only one?
   */
  bool single:1;
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
    finished(false),
    single(false) {
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
  return single;
}

inline libbirch::Label* libbirch::Any::getLabel() const {
  return (Label*)label;
}

inline void libbirch::Any::freeze() {
  if (!frozen) {
    frozen = true;
    auto nshared = numShared();
    single = nshared <= 1u && numWeak() <= 1u;
    if (nshared > 0u) {
      //doFreeze_();
    }
  }
}

inline void libbirch::Any::thaw(Label* label) {
  this->label = (intptr_t)label;
  frozen = false;
  finished = false;
  single = false;
  //doThaw_(label);
}
