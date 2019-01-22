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
   * If this object is frozen, return the object to which it should forward,
   * otherwise `this`.
   */
  Any* getForward();

  /**
   * If this object is frozen, and the object to which it should forward has
   * already been created, return that object, otherwise `this`.
   */
  Any* pullForward();

  /**
   * Get the memo responsible for the creation of this object.
   */
  Memo* getContext();

  /**
   * Is the object frozen?
   */
  bool isFrozen() const;

  /**
   * Freeze this object (as a result of it being lazily cloned).
   */
  void freeze();

protected:
  /**
   * Perform the actual freeze of the object. This is overwritten by derived
   * classes. The non-virtual freeze() handles thread safety so that this
   * need not.
   */
  virtual void doFreeze();

  /**
   * Memo responsible for the creation of this object.
   */
  WeakPtr<Memo> context;

  /**
   * If frozen, object to which to forward. This must be thread safe, and
   * so an atomic raw pointer is used, with manual shared reference count
   * maintenance.
   */
  std::atomic<Any*> forward;

  /**
   * Freeze count. This is 0 if the object is not frozen, a thread id plus
   * on if the object is in the process of being frozen (with the id that of
   * the thread doing so), and the number of threads plus one if the process
   * of being frozen is complete.
   */
  std::atomic<unsigned> freezeCount;
};
}
