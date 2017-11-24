/**
 * @file
 */
#pragma once

#include "bi/common/Location.hpp"

namespace bi {
/**
 * Object with a location within a file being parsed.
 *
 * @ingroup birch_common
 */
class Located {
public:
  /**
   * Constructor.
   *
   * @param loc Location.
   */
  Located(Location* loc = nullptr);

  /**
   * Destructor.
   */
  virtual ~Located() = 0;

  /**
   * Location.
   */
  Location* loc;
};
}
