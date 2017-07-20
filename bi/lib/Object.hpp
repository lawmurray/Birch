/**
 * @file
 */
#pragma once

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
      users(0),
      local(false) {
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
  virtual Object* clone() const;

  /**
   * Recursively replace one address with another.
   *
   * @param from Old address.
   * @param to New address.
   *
   * @return The new address of this object, which will be the same as the
   * old address if no replacements are necessary.
   *
   * If a replacement is made anywhere below this object, it will need to be
   * copied, and the return object will be different to this object,
   * otherwise it will be the same as this object.
   */
  virtual Object* replace(const void* from, const void* to);

  /**
   * Indicate that a(nother) coroutine is using this object.
   */
  void use() {
    ++users;
  }

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
    return local;
  }

  /**
   * Set whether this object is coroutine local.
   */
  void setLocal(const bool local = true) {
    this->local = local;
  }

private:
  /**
   * Number of coroutines using this object.
   */
  size_t users;

  /**
   * Is this object coroutine-local?
   */
  bool local;
};
}
