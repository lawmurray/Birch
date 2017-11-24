/**
 * @file
 */
#pragma once

#include "bi/common/Name.hpp"

namespace bi {
/**
 * Named object.
 *
 * @ingroup birch_common
 */
class Named {
public:
  /**
   * Empty constructor.
   */
  Named();

  /**
   * Constructor.
   *
   * @param name Name.
   */
  Named(Name* name);

  /**
   * Destructor.
   */
  virtual ~Named() = 0;

  /**
   * Name.
   */
  Name* name;
};
}
