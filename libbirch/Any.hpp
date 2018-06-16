/**
 * @file
 */
#pragma once

namespace bi {
template<class T> class SharedPointer;

/**
 * Base class for all class types.
 *
 * @ingroup libbirch
 */
class Any: public std::enable_shared_from_this<Any> {
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
  virtual std::shared_ptr<Any> clone() const;

  /**
   * Get the object world.
   */
  World* getWorld();

  /**
   * Create a shared pointer from this object.
   */
  template<class T>
  SharedPointer<T> shared_from_this() {
    auto ptr = enable_shared_from_this<Any>::shared_from_this();
    return SharedPointer<T>(std::static_pointer_cast<T>(ptr));
  }

protected:
  /**
   * The world to which this object belongs.
   */
  World* world;
};
}

#include "libbirch/PowerPoolAllocator.hpp"
#include "libbirch/global.hpp"

inline bi::Any::Any() : world(fiberWorld) {
  //
}

inline bi::Any::Any(const Any& o) : world(fiberWorld) {
  //
}

inline bi::Any::~Any() {
  //
}

inline std::shared_ptr<bi::Any> bi::Any::clone() const {
  return std::allocate_shared<Any>(PowerPoolAllocator<Any>(), *this);
}

inline bi::World* bi::Any::getWorld() {
  return world;
}
