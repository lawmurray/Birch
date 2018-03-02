/**
 * @file
 */
#pragma once

#include "libbirch/global.hpp"

namespace bi {
/**
 * Auxiliary class for switching the current world. This may be used either
 * on the stack to switch the world for the lifetime of a temporary variable:
 *
 *     Enter enter(world);
 *
 * or inherited from a class to switch the world during initialization,
 * explicitly calling exit() to restore the previous world.
 *
 *     class A : public Enter {
 *       A() : ..., Enter(world), ... {
 *         exit();
 *       }
 *     }
 */
class Enter {
public:
  /**
   * Constructor.
   *
   * @param world The world to enter.
   */
  Enter(World* world) :
      prevWorld(world) {
    std::swap(prevWorld, fiberWorld);
  }

  /**
   * Destructor.
   */
  ~Enter() {
    exit();
  }

  /**
   * Exit.
   */
  void exit() {
    if (prevWorld) {
      std::swap(prevWorld, fiberWorld);
      prevWorld = nullptr;
    }
  }

private:
  /**
   * The previous world.
   */
  World* prevWorld;
};
}
