/**
 * @file
 */
#pragma once

#include "bi/lib/global.hpp"
#include "bi/lib/AllocationMap.hpp"

#include <cassert>
#include <cstdint>

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
  Any() : gen(fiberAllocationMap->gen) {
    //
  }

  /**
   * Copy constructor.
   */
  Any(const Any& o) :
      gen(fiberAllocationMap->gen) {
    //
  }

  /**
   * Destructor.
   */
  virtual ~Any() {
    //
  }

  /**
   * Clone the object. This is a shallow clone with fiber usage counts of
   * member attributes incremented, deferring their cloning until they are
   * used.
   */
  virtual Any* clone() = 0;

  /**
   * Is this object (possibly) shared?
   */
  size_t isShared() const {
    return gen < fiberAllocationMap->gen;
  }

  /**
   * Get a smart pointer to this object.
   */
  template<class T>
  Pointer<T> pointer_from_this();
  template<class T>
  Pointer<const T> pointer_from_this() const;

//private:
  /**
   * Generation in which this object was created.
   */
  size_t gen;
};
}

#include "bi/lib/Pointer.hpp"

template<class T>
bi::Pointer<T> bi::Any::pointer_from_this() {
  return Pointer<T>(static_cast<T*>(this));
}

template<class T>
bi::Pointer<const T> bi::Any::pointer_from_this() const {
  return Pointer<const T>(static_cast<T* const>(this));
}
