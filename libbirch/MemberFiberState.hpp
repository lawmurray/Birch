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
 * @tparam ObjectType Type of owning object.
 * @tparam ArgumentType Type of arguments to fiber.
 * @tparam LocalType Type of local variables of fiber.
 */
template<class YieldType, class ObjectType, class ArgumentType, class LocalType>
class MemberFiberState: public FiberState<YieldType> {
public:
  using this_type = MemberFiberState<YieldType,ObjectType,ArgumentType,LocalType>;
  using super_type = FiberState<YieldType>;

  /**
   * Constructor.
   *
   * @param label Initial label.
   * @param nlabels Number of labels.
   * @param object Owning object.
   * @param arg Arguments to fiber call on owning object.
   */
  MemberFiberState(const int label, const int nlabels,
      const SharedPointer<ObjectType>& object,
      const std::shared_ptr<ArgumentType>& arg) :
      super_type(label, nlabels),
      object(object),
      arg(arg),
      local(nullptr, object.getWorld()),
      value(nullptr, object.getWorld()) {
    Enter enter(object.getWorld());
    local = std::make_shared<LocalType>();
    value = std::make_shared<YieldType>();
  }

  /**
   * Copy constructor.
   */
  MemberFiberState(const this_type& o) :
      super_type(o),
      object(o.object),
      arg(o.arg),
      local(nullptr, object.getWorld()),
      value(nullptr, object.getWorld()) {
    EnterClone enter(object.getWorld());
    local = std::make_shared<LocalType>(*o.local);
    value = std::make_shared<YieldType>(*o.value);
  }

  virtual const std::shared_ptr<World>& getWorld() {
    return object.getWorld();
  }

  virtual YieldType get() {
    return *value;
  }

  ObjectType* self() {
    return object->self();
  }

  typename ObjectType::super_type* super() {
    return object->super();
  }

  SharedPointer<ObjectType> shared_self() {
    return object->shared_self();
  }

  SharedPointer<typename ObjectType::super_type> shared_super() {
    return object->shared_super();
  }

protected:
  /**
   * Owning object.
   */
  SharedPointer<ObjectType> object;

  /**
   * Arguments.
   */
  SharedPointer<ArgumentType> arg;

  /**
   * Local variables.
   */
  SharedPointer<LocalType> local;

  /**
   * Yield value.
   */
  SharedPointer<YieldType> value;
};
}
