/**
 * @file
 */
#pragma once

namespace bi {
class FiberWorld;

/**
 * Base class for all class types.
 *
 * @ingroup libbirch
 */
class Any {
public:
  /**
   * Constructor.
   */
  Any();

  /**
   * Clone the object. This is a shallow clone with fiber usage counts of
   * member attributes incremented, deferring their cloning until they are
   * used.
   */
  virtual Any* clone() = 0;

  /**
   * The world in which this object exists.
   */
  FiberWorld* world;
};
}
