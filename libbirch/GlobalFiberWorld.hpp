/**
 * @file
 */
#pragma once

namespace bi {
/**
 * Auxiliary class for GlobalFiberState to ensure correct order of
 * initialization.
 */
class GlobalFiberWorld {
public:
  /**
   * Constructor.
   */
  GlobalFiberWorld() :
      world(std::make_shared<World>()) {
    //
  }

  /**
   * Constructor.
   */
  GlobalFiberWorld(const std::shared_ptr<World>& parent) :
      world(std::make_shared<World>(parent)) {
    //
  }

protected:
  /**
   * Fiber world.
   */
  std::shared_ptr<World> world;
};
}
