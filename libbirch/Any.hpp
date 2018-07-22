/**
 * @file
 */
#pragma once

#include "libbirch/Counted.hpp"

namespace bi {
template<class T> class SharedCOW;

/**
 * Base class for all class types.
 *
 * @ingroup libbirch
 */
class Any : public Counted {
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
   * Get the object world.
   */
  World* getWorld();

protected:
  /**
   * The world to which this object belongs.
   */
  World* world;
};
}

#include "libbirch/global.hpp"

inline bi::Any::Any() :
    world(fiberWorld) {
  //
}

inline bi::Any::Any(const Any& o) :
    world(fiberWorld) {
  //
}

inline bi::Any::~Any() {
  //
}

inline bi::Any* bi::Any::clone() const {
  return bi::construct<Any>(*this);
}

inline void bi::Any::destroy() {
  this->size = sizeof(*this);
  this->~Any();
}

inline bi::World* bi::Any::getWorld() {
  return world;
}
