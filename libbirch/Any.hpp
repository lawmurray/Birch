/**
 * @file
 */
#pragma once

#include "libbirch/SharedPointer.hpp"
#include "libbirch/WeakPointer.hpp"

namespace bi {
/**
 * Base class for all class types.
 *
 * @ingroup libbirch
 */
class Any {
  friend class Allocation;
public:
  /**
   * Default constructor.
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
  virtual Any* clone();

protected:
  /**
   * Create a shared pointer from this object.
   */
  template<class T>
  SharedPointer<T> shared_from_this() const;

private:
  /**
   * Weak pointer to self, used by shared_from_this() to construct a shared
   * pointer to this object.
   */
  WeakPointer<Any> ptr;
};
}

template<class T>
bi::SharedPointer<T> bi::Any::shared_from_this() const {
#ifdef NDEBUG
  return ptr.template static_pointer_cast<T>();
#else
  auto result = ptr.template dynamic_pointer_cast<T>();
  assert(result.query());
  assert(result.get() == this);
  return result;
#endif
}
