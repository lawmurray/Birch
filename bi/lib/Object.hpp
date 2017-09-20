/**
 * @file
 */
#pragma once

#include <cassert>

namespace bi {
template<class T> class Pointer;
/**
 * Base class for all class types. Includes functionality for sharing objects
 * between fibers with copy-on-write semantics.
 *
 * @ingroup library
 */
class Object {
public:
  /**
   * Constructor.
   */
  Object() :
      users(1),
      index(-1) {
    //
  }

  /**
   * Copy constructor.
   */
  Object(const Object& o) :
      users(1),
      index(o.index) {
    //
  }

  /**
   * Destructor.
   */
  virtual ~Object() {
    //
  }

  /**
   * Clone the object. This is a shallow clone with fiber usage counts of
   * member attributes incremented, deferring their cloning until they are
   * used.
   */
  virtual Object* clone() = 0;

  /**
   * Indicate that a(nother) fiber is using this object.
   */
  void use() {
    ++users;
  }

  /**
   * Indicate that a fiber is no longer using this object.
   */
  void disuse() {
    assert(users > 0);
    --users;
  }

  /**
   * Is this object being shared by two or more fibers?
   */
  bool isShared() const {
    return users > 1;
  }

  /**
   * Get the fiber-local heap index of the object.
   */
  size_t getIndex() const {
    return index;
  }

  /**
   * Set the fiber-local heap index of the object.
   */
  void setIndex(const size_t index) {
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
   * Number of fibers referencing this object.
   */
  size_t users;

  /**
   * For a fiber-local pointer, the index of the heap allocation,
   * otherwise -1.
   */
  intptr_t index;
};
}

#include "bi/lib/Pointer.hpp"

template<class T>
bi::Pointer<T> bi::Object::pointer_from_this() {
  return Pointer<T>(static_cast<T*>(this), index);
}

template<class T>
bi::Pointer<const T> bi::Object::pointer_from_this() const {
  return Pointer<const T>(static_cast<T* const>(this), index);
}
