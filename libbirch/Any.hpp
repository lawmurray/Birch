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
  return result;
#endif
}
