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
class GlobalFiberState: public FiberState<YieldType>,
    protected ArgumentType,
    public GlobalFiberWorld,
    protected Enter,
    protected LocalType {
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
      FiberState<YieldType>(label, nlabels),
      ArgumentType { args... },
      GlobalFiberWorld(),  // creates fiber's world
      Enter(getWorld()),  // enters fiber's world
      LocalType() {
    exit();  // exits fiber's world
  }

  /**
   * Copy constructor.
   */
  GlobalFiberState(
      const GlobalFiberState<YieldType,ArgumentType,LocalType>& o) :
      FiberState<YieldType>(o),
      ArgumentType(o),
      GlobalFiberWorld(o.world),  // creates fiber's world
      Enter(getWorld()),  // enters fiber's world
      LocalType(o) {
    exit();  // exits fiber's world
  }

  virtual void destroy() {
    this->size = sizeof(*this);
    this->~GlobalFiberState();
  }

  /**
   * Get the world to which the fiber state belongs.
   */
  virtual World* getWorld() {
    return world.get();
  }

  virtual YieldType& get() {
    return value;
  }

protected:
  /**
   * Yield value.
   */
  YieldType value;
};
}
