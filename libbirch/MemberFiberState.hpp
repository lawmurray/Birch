/**
 * @file
 */
#pragma once

#include "libbirch/FiberState.hpp"
#include "libbirch/MemberFiberWorld.hpp"
#include "libbirch/World.hpp"
#include "libbirch/Enter.hpp"

namespace bi {
/**
 * State of a global fiber.
 *
 * @ingroup libbirch
 *
 * @tparam YieldType Type of yield value.
 * @tparam ObjectType Type of owning object.
 * @tparam ArgumentType Type of arguments to fiber.
 * @tparam LocalType Type of local variables of fiber.
 */
template<class YieldType, class ObjectType, class ArgumentType,
    class LocalType>
class MemberFiberState:
    protected ArgumentType,
    public MemberFiberWorld<ObjectType>,
    protected Enter,
    protected LocalType,
    public FiberState<YieldType> {
public:
  /**
   * Constructor.
   *
   * @tparam Args... Argument types.
   *
   * @param label Initial label.
   * @param nlabels Number of labels.
   * @param object Owning object.
   * @param args Arguments to fiber call.
   */
  template<class ... Args>
  MemberFiberState(const int label, const int nlabels,
      const SharedCOW<ObjectType>& object, Args ... args) :
      ArgumentType { args... },
      MemberFiberWorld<ObjectType>(object),
      Enter(getWorld()),  // enters owning object's world
      LocalType(),
      FiberState<YieldType>(label, nlabels) {
    exit();  // exits owning object's world
  }

  /**
   * Copy constructor.
   */
  MemberFiberState(
      const MemberFiberState<YieldType,ObjectType,ArgumentType,LocalType>& o) :
      ArgumentType(o),
      MemberFiberWorld<ObjectType>(o),
      Enter(getWorld()),  // enters owning object's world
      LocalType(o),
      FiberState<YieldType>(o) {
    exit();  // exits owning object's world
  }

  /**
   * Deallocate.
   */
  virtual void deallocate() {
    bi::deallocate(this, sizeof(this));
  }

  /**
   * Get the world to which the fiber state belongs.
   */
  virtual World* getWorld() {
    return this->object->getWorld();
  }

  auto self() {
    return this->object->self();
  }
};
}
