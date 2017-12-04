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
   * Constructor.
   */
  Any();

  /**
   * Destructor.
   */
  virtual ~Any() = 0;

  /**
   * Clone the object. This is a shallow clone with fiber usage counts of
   * member attributes incremented, deferring their cloning until they are
   * used.
   */
  virtual Any* clone() = 0;

  /**
   * Create a pointer for this object.
   */
  Pointer<Any> pointer_from_this();
};
}
