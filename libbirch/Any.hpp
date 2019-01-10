/**
 * @file
 */
#pragma once

#include "libbirch/config.hpp"
#include "libbirch/Counted.hpp"
#include "libbirch/List.hpp"

namespace bi {
template<class T> class SharedCOW;

/**
 * Base for all class types.
 *
 * @ingroup libbirch
 */
class Any: public Counted {
public:
  using class_type = Any;
  using this_type = Any;

protected:
  /**
   * Constructor.
   */
  Any();

  /**
   * Copy constructor.
   */
  Any(const Any& o);

  /**
   * Destructor.
   */
  virtual ~Any();

  /**
   * Copy assignment operator.
   */
  Any& operator=(const Any&) = delete;

public:
  STANDARD_CREATE_FUNCTION
  STANDARD_EMPLACE_FUNCTION
  STANDARD_CLONE_FUNCTION
  STANDARD_DESTROY_FUNCTION

  /**
   * Is the object frozen?
   */
  bool isFrozen() const;

  /**
   * Freeze this object (as a result of it being lazily cloned).
   */
  virtual void freeze();

  /**
   * Get the object to which to forward if this object is frozen, otherwise
   * `this`.
   */
  Any* getForward();

  /**
   * Get the memo responsible for the creation of this object.
   */
  Memo* getContext();

protected:
  /**
   * Memo responsible for the creation of this object.
   */
  WeakPtr<Memo> context;

  /**
   * If frozen, object to which to forward.
   */
  SharedPtr<Any> forward;

  /**
   * Is this object frozen (as a result of being lazily cloned)?
   */
  bool frozen;
};
}
