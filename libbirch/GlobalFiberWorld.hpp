/**
 * @file
 */
#pragma once

namespace bi {
/**
 * Auxiliary class for GlobalFiberState to ensure correct order of
 * initialization.
 *
 * @ingroup libbirch
 */
class GlobalFiberWorld {
public:
  /**
   * Constructor.
   */
  GlobalFiberWorld() :
      world(bi::construct<World>()) {
    //
  }

  /**
   * Constructor.
   */
  GlobalFiberWorld(const SharedPtr<World> cloneSource) :
      world(bi::construct<World>(cloneSource)) {
    //
  }

  /**
   * Destructor.
   */
  virtual ~GlobalFiberWorld() {
    //
  }

protected:
  /**
   * Fiber world.
   */
  SharedPtr<World> world;
};
}
