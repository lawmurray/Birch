/**
 * @file
 */
#pragma once

#include "libbirch/SharedPointer.hpp"

namespace bi {
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
  std::weak_ptr<World> getWorld();

protected:
  /**
   * Create a shared pointer from this object.
   */
  template<class T>
  SharedPointer<T> shared_from_this();

  /**
   * The world to which this object belongs.
   */
  std::weak_ptr<World> world;
};
}

template<class T>
bi::SharedPointer<T> bi::Any::shared_from_this() {
  auto ptr = enable_shared_from_this<Any>::shared_from_this();
#ifndef NDEBUG
  return std::dynamic_pointer_cast<T>(ptr);
#else
  return std::static_pointer_cast<T>(ptr);
#endif
}
