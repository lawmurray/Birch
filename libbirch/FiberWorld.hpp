/**
 * @file
 */
#pragma once

#include "libbirch/global.hpp"

namespace bi {
/**
 * World of a fiber.
 *
 * @ingroup libbirch
 */
class FiberWorld {
public:
  /**
   * Constructor.
   */
  FiberWorld(const FiberWorld* parent = fiberWorld);

  /**
   * Id of this world.
   */
  const size_t id;

  /**
   * Parent world.
   */
  const FiberWorld* parent;

private:
  /**
   * The number of worlds created so far.
   */
  static size_t nworlds;
};
}
