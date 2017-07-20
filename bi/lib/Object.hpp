/**
 * @file
 */
#pragma once

#include "bi/lib/Coroutine.hpp"

#include <cassert>

namespace bi {
/**
 * Base class for all class types. Includes functionality for sharing objects
 * between coroutines with copy-on-write semantics.
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
   * Clone the object. This is a shallow clone with coroutine usage counts
   * of member attributes incremented, deferring their cloning until they are
   * used.
   */
  virtual Object* clone() = 0;

  /**
   * Indicate that a coroutine is no longer using this object.
   */
  void disuse() {
    assert(users > 0);
    --users;
  }

  /**
   * Is this object being shared by two or more coroutines?
   */
  bool isShared() const {
    return users > 1;
  }

  /**
   * Is this object coroutine-local?
   */
  bool isLocal() const {
    return index >= 0;
  }

  /**
   * Get a smart pointer to this object.
   */
  template<class T>
  Pointer<T> pointer_from_this() {
    return Pointer<T>(static_cast<T*>(this), index);
  }
  template<class T>
  Pointer<const T> pointer_from_this() const {
    return Pointer<const T>(static_cast<T* const>(this), index);
  }

private:
  /**
   * Number of coroutines using this object.
   */
  size_t users;

  /**
   * For a coroutine-local pointer, the index of the heap allocation,
   * otherwise -1.
   */
  size_t index;
};
}
