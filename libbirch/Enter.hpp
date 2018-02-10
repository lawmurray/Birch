/**
 * @file
 */
#pragma once

#include "libbirch/global.hpp"

namespace bi {
/**
 * Temporarily enter a new world. The object loads the given world on
 * construction, and restores the previous world on destruction.
 */
class Enter {
public:
  /**
   * Constructor.
   *
   * @param world The world to enter.
   */
  Enter(const std::shared_ptr<World>& world) : prevWorld(fiberWorld) {
    fiberWorld = world;
  }

  /**
   * Destructor.
   */
  ~Enter() {
    fiberWorld = prevWorld;
  }

private:
  /**
   * The previous world.
   */
  std::shared_ptr<World> prevWorld;
};
}
