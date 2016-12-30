/**
 * @file
 */
#pragma once

#include "bi/common/Location.hpp"
#include "bi/primitive/shared_ptr.hpp"

namespace bi {
/**
 * Object with a location within a file being parsed.
 *
 * @ingroup compiler_common
 */
class Located {
public:
  /**
   * Constructor.
   *
   * @param loc Location.
   */
  Located(shared_ptr<Location> loc = nullptr);

  /**
   * Destructor.
   */
  virtual ~Located() = 0;

  /**
   * Location.
   */
  shared_ptr<Location> loc;
};
}
