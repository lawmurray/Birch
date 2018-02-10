/**
 * @file
 */
#pragma once

namespace bi {
/**
 * State of a member fiber.
 *
 * @ingroup libbirch
 *
 * @tparam YieldType Yield type.
 * @tparam ObjectType Object type.
 */
template<class YieldType, class ObjectType>
class MemberFiberState: public FiberState<YieldType> {
public:
  using this_type = ObjectType;
  using super_type = typename ObjectType::super_type;

  /**
   * Constructor.
   */
  MemberFiberState(SharedPointer<ObjectType> o, const int label = 0,
      const int nlabels = 0) :
      FiberState<YieldType>(label, nlabels),
      o(o) {
    //
  }

  /**
   * Destructor.
   */
  virtual ~MemberFiberState() {
    //
  }

  const std::shared_ptr<World>& getWorld() {
    return o->getWorld();
  }

  this_type* self() {
    return o->self();
  }

  super_type* super() {
    return o->super();
  }

  SharedPointer<this_type> shared_self() {
    return o->shared_self();
  }

  SharedPointer<super_type> shared_super() {
    return o->shared_super();
  }

protected:
  /**
   * Object.
   */
  SharedPointer<ObjectType> o;
};
}
