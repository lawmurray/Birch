/**
 * @file
 */
#pragma once

#include "libbirch/World.hpp"

namespace bi {
/**
 * State of a global fiber.
 *
 * @ingroup libbirch
 *
 * @tparam YieldType Yield type.
 */
template<class YieldType>
class GlobalFiberState: public FiberState<YieldType> {
public:
  /**
   * Constructor.
   */
  GlobalFiberState(const int label = 0, const int nlabels = 0) :
      FiberState<YieldType>(label, nlabels),
      world(std::make_shared<World>(fiberWorld, nullptr)) {
    //
  }

  /**
   * Copy constructor.
   */
  GlobalFiberState(const GlobalFiberState<YieldType>& o) :
      FiberState<YieldType>(o),
      world(std::make_shared<World>(fiberWorld, o.world)) {
    //
  }

  /**
   * Destructor.
   */
  virtual ~GlobalFiberState() {
    //
  }

  const std::shared_ptr<World>& getWorld() {
    return world;
  }

protected:
  /**
   * World.
   */
  std::shared_ptr<World> world;
};
}
