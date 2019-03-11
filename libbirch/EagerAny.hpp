/**
 * @file
 */
#pragma once
#if !ENABLE_LAZY_DEEP_CLONE

#include "libbirch/Counted.hpp"

namespace libbirch {
/**
 * Base for all class types when eager deep clone is used.
 *
 * @ingroup libbirch
 */
class EagerAny: public Counted {
public:
  using class_type_ = EagerAny;
  using this_type_ = EagerAny;

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
  libbirch_create_function_
  libbirch_emplace_function_
  libbirch_clone_function_
  libbirch_destroy_function_
};
}

#endif
