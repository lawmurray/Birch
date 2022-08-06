/**
 * @file
 */
#pragma once

#include "src/common/Name.hpp"

namespace birch {
/**
 * Named object.
 *
 * @ingroup common
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
   * Name.
   */
  Name* name;
};
}
