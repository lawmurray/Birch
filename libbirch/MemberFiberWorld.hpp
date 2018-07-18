/**
 * @file
 */
#pragma once

namespace bi {
/**
 * Auxiliary class for MemberFiberState to ensure correct order of
 * initialization.
 *
 * @ingroup libbirch
 *
 * @tparam ObjectType Type of owning object.
 */
template<class ObjectType>
class MemberFiberWorld {
public:
  /**
   * Constructor.
   */
  MemberFiberWorld(const SharedCOW<ObjectType>& object) :
      object(object) {
    //
  }

  /**
   * Destructor.
   */
  virtual ~MemberFiberWorld() {
    //
  }

protected:
  /**
   * Owning object.
   */
  SharedCOW<ObjectType> object;
};
}
