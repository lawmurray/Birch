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
   * Get the memo responsible for the creation of this object.
   */
  Memo* getContext();

  /**
   * Record a clone for later cleanup purposes.
   */
  void recordClone(Any* o);

protected:
  /**
   * Memo responsible for the creation of this object.
   */
  WeakPtr<Memo> context;

  #if USE_LAZY_DEEP_CLONE_FORWARD_CLEAN
  /**
   * Clones of this, kept for later cleanup.
   */
  List clones;
  #endif
};
}
