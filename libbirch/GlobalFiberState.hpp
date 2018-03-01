/**
 * @file
 */
#pragma once

#include "libbirch/FiberState.hpp"
#include "libbirch/GlobalFiberWorld.hpp"
#include "libbirch/World.hpp"
#include "libbirch/Enter.hpp"

namespace bi {
/**
 * State of a global fiber.
 *
 * @ingroup libbirch
 *
 * @tparam YieldType Type of yield value.
 * @tparam ArgumentType Type of arguments to fiber.
 * @tparam LocalType Type of local variables of fiber.
 */
template<class YieldType, class ArgumentType, class LocalType>
class GlobalFiberState:
    public ArgumentType,
    public GlobalFiberWorld,
    public Enter,
    public LocalType,
    public FiberState<YieldType> {
public:
  /**
   * Constructor.
   *
   * @tparam Args... Argument types.
   *
   * @param label Initial label.
   * @param nlabels Number of labels.
   * @param args Arguments to fiber call.
   */
  template<class ... Args>
  GlobalFiberState(const int label, const int nlabels, Args ... args) :
      ArgumentType( { args... }),
      GlobalFiberWorld(),  // creates fiber's world
      Enter(world),  // enters fiber's world
      LocalType(),
      FiberState<YieldType>(label, nlabels) {
    exit();  // exits fiber's world
  }

  /**
   * Copy constructor.
   */
  GlobalFiberState(
      const GlobalFiberState<YieldType,ArgumentType,LocalType>& o) :
      ArgumentType(o),
      GlobalFiberWorld(o.world),  // creates fiber's world
      Enter(world),  // enters fiber's world
      LocalType(o),
      FiberState<YieldType>(o) {
    exit();  // exits fiber's world
  }

  virtual std::shared_ptr<World> getWorld() {
    return world;
  }
};
}
