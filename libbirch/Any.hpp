/**
 * @file
 */
#pragma once

#include "libbirch/Counted.hpp"

namespace bi {
template<class T> class SharedCOW;

/**
 * Base for all class types.
 *
 * @ingroup libbirch
 */
class Any: public Counted {
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
   * Clone the object.
   */
  virtual Any* clone() const;

  /**
   * Deallocate the memory for the object.
   */
  virtual void destroy();

  /**
   * Get the memo associated with the clone or construction of this object.
   */
  Memo* getMemo();

protected:
  /**
   * Memo associated with the clone or construction of this object.
   */
  Memo* memo;
};
}
