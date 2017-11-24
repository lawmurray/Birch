/**
 * @file
 */
#pragma once

namespace bi {
/**
 * Base class for all class types.
 *
 * @ingroup libbirch
 */
class Any {
public:
  /**
   * Clone the object. This is a shallow clone with fiber usage counts of
   * member attributes incremented, deferring their cloning until they are
   * used.
   */
  virtual Any* clone() = 0;
};
}
