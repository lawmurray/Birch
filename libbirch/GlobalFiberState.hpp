/**
 * @file
 */
#pragma once

#include "libbirch/FiberState.hpp"
#include "libbirch/World.hpp"
#include "libbirch/Enter.hpp"
#include "libbirch/EnterClone.hpp"

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
class GlobalFiberState: public FiberState<YieldType> {
public:
  using this_type = GlobalFiberState<YieldType,ArgumentType,LocalType>;
  using super_type = FiberState<YieldType>;

  /**
   * Constructor.
   *
   * @param label Initial label.
   * @param nlabels Number of labels.
   * @param arg Arguments to fiber call.
   */
  GlobalFiberState(const int label, const int nlabels,
      const std::shared_ptr<ArgumentType>& arg) :
      super_type(label, nlabels),
      arg(arg),
      world(std::make_shared<World>()),
      local(world),
      value(world) {
    Enter enter(world);
    local = std::make_shared<LocalType>();
    value = std::make_shared<YieldType>();
  }

  /**
   * Copy constructor.
   */
  GlobalFiberState(const this_type& o) :
      super_type(o),
      arg(o.arg),
      world(std::make_shared<World>(o.world)),
      local(world),
      value(world) {
    EnterClone enter(world);
    local.reset(new LocalType(*o.local));
    value.reset(new YieldType(*o.value));
  }

  virtual const std::shared_ptr<World>& getWorld() {
    return world;
  }

  virtual YieldType get() {
    return *value;
  }

protected:
  /* the order in which these are declared is important, in particular world
   * must be initialized before local and value, which will belong to that
   * world, while arg belongs to the caller world */
  /**
   * Arguments.
   */
  std::shared_ptr<ArgumentType> arg;

  /**
   * Local world.
   */
  std::shared_ptr<World> world;

  /**
   * Local variables.
   */
  std::shared_ptr<LocalType> local;

  /**
   * Local yield value.
   */
  std::shared_ptr<YieldType> value;
};
}
