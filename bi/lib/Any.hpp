/**
 * @file
 */
#pragma once

namespace bi {
template<class T> class Pointer;
/**
 * Base class for all class types. Includes functionality for sharing objects
 * between fibers with copy-on-write semantics.
 *
 * @ingroup library
 */
class Any {
public:
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
   * Clone the object. This is a shallow clone with fiber usage counts of
   * member attributes incremented, deferring their cloning until they are
   * used.
   */
  virtual Any* clone() = 0;

  /**
   * Is this object (possibly) shared?
   */
  bool isShared() const;

//private:
  /**
   * Generation in which this object was created.
   */
  size_t gen;
};
}
