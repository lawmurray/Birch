/**
 * @file
 */
#pragma once

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
  Any() {
    //
  }

  /**
   * Copy constructor.
   */
  Any(const Any& o) :
      gen(o.gen),
      index(o.index) {
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
   * Get the fiber generation of the object.
   */
  size_t getGen() const {
    return gen;
  }

  /**
   * Set the fiber generation of the object.
   */
  void setGen(const size_t gen) {
    this->gen = gen;
  }

  /**
   * Get the fiber-local heap index of the object.
   */
  intptr_t getIndex() const {
    return index;
  }

  /**
   * Set the fiber-local heap index of the object.
   */
  void setIndex(const intptr_t index) {
    this->index = index;
  }

  /**
   * Get a smart pointer to this object.
   */
  template<class T>
  Pointer<T> pointer_from_this();
  template<class T>
  Pointer<const T> pointer_from_this() const;

private:
  /**
   * Fiber generation in which this object was created.
   */
  size_t gen;

  /**
   * Index of the heap allocation.
   */
  intptr_t index;
};
}

#include "bi/lib/Pointer.hpp"

template<class T>
bi::Pointer<T> bi::Any::pointer_from_this() {
  return Pointer<T>(static_cast<T*>(this), index);
}

template<class T>
bi::Pointer<const T> bi::Any::pointer_from_this() const {
  return Pointer<const T>(static_cast<T* const>(this), index);
}
