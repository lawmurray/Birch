/**
 * @file
 */
#pragma once
#if !USE_LAZY_DEEP_CLONE

#include "libbirch/config.hpp"
#include "libbirch/Counted.hpp"

namespace bi {
/**
 * Base for all class types when eager deep clone is used.
 *
 * @ingroup libbirch
 */
class EagerAny: public Counted {
public:
  using class_type = EagerAny;
  using this_type = EagerAny;

protected:
  /**
   * Constructor.
   */
  EagerAny();

  /**
   * Copy constructor.
   */
  EagerAny(const EagerAny& o);

  /**
   * Destructor.
   */
  virtual ~EagerAny();

  /**
   * Copy assignment operator.
   */
  EagerAny& operator=(const EagerAny&) = delete;

public:
  STANDARD_CREATE_FUNCTION
  STANDARD_EMPLACE_FUNCTION
  STANDARD_CLONE_FUNCTION
  STANDARD_DESTROY_FUNCTION
};
}

#endif
