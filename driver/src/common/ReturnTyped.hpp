/**
 * @file
 */
#pragma once

#include "src/type/Type.hpp"
#include "src/type/EmptyType.hpp"

namespace birch {
/**
 * Function or operator with return type.
 *
 * @ingroup common
 */
class ReturnTyped {
public:
  /**
   * Constructor.
   *
   * @param returnType Return type.
   */
  ReturnTyped(Type* returnType);

  /**
   * Return type.
   */
  Type* returnType;
};
}
