/**
 * @file
 */
#pragma once

#include "libbirch/Enter.hpp"

namespace bi {
/**
 * Enter a world for the purposes of cloning an object.
 */
class EnterClone: public Enter {
public:
  /**
   * Constructor.
   */
  EnterClone(const std::shared_ptr<World>& world) :
      Enter(world),
      prevCloning(fiberCloning) {
    fiberCloning = true;
  }

  /**
   * Destructor.
   */
  ~EnterClone() {
    fiberCloning = prevCloning;
  }

private:
  /**
   * The previous flag.
   */
  bool prevCloning;
};
}
