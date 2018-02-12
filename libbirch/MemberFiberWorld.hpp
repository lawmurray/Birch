/**
 * @file
 */
#pragma once

namespace bi {
/**
 * Auxiliary class for MemberFiberState to ensure correct order of
 * initialization.
 *
 * @tparam ObjectType Type of owning object.
 */
template<class ObjectType>
class MemberFiberWorld {
public:
  /**
   * Constructor.
   */
  MemberFiberWorld(const SharedPointer<ObjectType>& object) :
      object(object) {
    //
  }

protected:
  /**
   * Owning object.
   */
  SharedPointer<ObjectType> object;
};
}
