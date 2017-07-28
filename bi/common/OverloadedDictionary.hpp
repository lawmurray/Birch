/**
 * @file
 */
#pragma once

#include "bi/common/Dictionary.hpp"
#include "bi/common/Overloaded.hpp"

namespace bi {
/**
 * Dictionary for overloaded functions, operators etc.
 *
 * @ingroup compiler_common
 *
 * @tparam ObjectType Type of objects.
 */
template<class ObjectType>
class OverloadedDictionary: public Dictionary<Overloaded<ObjectType>> {
public:
  /**
   * Destructor.
   */
  ~OverloadedDictionary();

  /**
   * Add object.
   *
   * @param o The object.
   */
  void add(ObjectType* o);
};
}
