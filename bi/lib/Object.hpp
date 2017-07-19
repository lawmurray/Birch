/**
 * @file
 */
#pragma once

#include <cassert>

namespace bi {
/**
 * Base of all class types.
 */
class Object {
public:
  /**
   * Constructor.
   */
  Object() :
      users(0),
      index(-1) {
    //
  }

  /**
   * Destructor.
   */
  virtual ~Object() {
    assert(users == 0);
  }

  /**
   * Copy the object.
   */
  virtual Object* copy() const = 0;

  /**
   * Mark the object.
   */
  virtual void mark() const = 0;

  /**
   * Indicate that a(nother) coroutine is using this object.
   *
   * @param index The index of this object in the coroutine's local heap.
   */
  void use(const int index) {
    ++users;
    if (users == 1) {
      this->index = index;
    } else {
      assert(this->index == index);
    }
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

protected:
  /**
   * Create a pointer to this object. This function should be used instead of
   * @c this to obtain a correctly constructed pointer for global or
   * coroutine-local heap allocation, as appropriate for the object. It
   * behaves similarly to @c shared_from_this() in the STL.
   */
  template<class DerivedType>
  Pointer<DerivedType> pointer_from_this() {
    if (index >= 0) {
      return Pointer<DerivedType>(index);
    } else {
      return Pointer<DerivedType>(dynamic_cast<DerivedType*>(this));
    }
  }
  template<class DerivedType>
  Pointer<const DerivedType> pointer_from_this() const {
    if (index >= 0) {
      return Pointer<const DerivedType>(index);
    } else {
      return Pointer<const DerivedType>(
          dynamic_cast<DerivedType* const >(this));
    }
  }

private:
  /**
   * Number of coroutines using this object.
   */
  size_t users;

  /**
   * Position of this object in coroutine-local heaps, or -1 if on the global
   * heap. Note that an object can only be in use by more than one coroutine
   * via the copying of coroutines, so that it must have the same position in
   * all coroutine-local heaps in which it exists.
   */
  long index;
};
}
