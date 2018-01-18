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
  World* getWorld();

protected:
  /**
   * Create a shared pointer from this object.
   */
  template<class T>
  SharedPointer<T> shared_from_this();

  /**
   * Create a shared pointer from this object.
   */
  template<class T>
  SharedPointer<const T> shared_from_this() const;

  /**
   * The world to which this object belongs.
   */
  World* world;
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

template<class T>
bi::SharedPointer<const T> bi::Any::shared_from_this() const {
  auto ptr = enable_shared_from_this<Any>::shared_from_this();
#ifndef NDEBUG
  return std::dynamic_pointer_cast<const T>(ptr);
#else
  return std::static_pointer_cast<const T>(ptr);
#endif
}
