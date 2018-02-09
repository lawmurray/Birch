/**
 * @file
 */
#pragma once

namespace bi {
/**
 * Wraps a pointer during dereferencing. The life of the object begins as the
 * pointer is dereferenced, and ends after the operation on the object
 * concludes (e.g. an assignment to a member variable, or calling a member
 * function). The constructor and destructor are used to enter and exit the
 * world of the object, respectively.
 *
 * @tparam T Object type.
 */
template<class T>
class Wrapper {
public:
  /**
   * Constructor.
   */
  Wrapper(T* o) : o(o), prevWorld(fiberWorld) {
    /* enter the object's world */
    fiberWorld = o->getWorld();
  }

  /**
   * Destructor.
   */
  ~Wrapper() {
    /* exit the object's world, restoring the previously active world */
    fiberWorld = prevWorld;
  }

  /**
   * Dereference.
   */
  T& operator*() {
    return *o;
  }

  /**
   * Member access.
   */
  T* operator->() {
    return o;
  }

private:
  /**
   * The wrapped object.
   */
  T* o;

  /**
   * The active world at the time of construction.
   */
  std::shared_ptr<World> prevWorld;
};
}
