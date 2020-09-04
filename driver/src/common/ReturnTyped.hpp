/**
 * @file
 */
#pragma once

#include "src/type/Type.hpp"
#include "src/type/EmptyType.hpp"

namespace birch {
/**
 * Function, fiber or operator with return type.
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
   * Destructor.
   */
  virtual ~ReturnTyped() = 0;

  /**
   * Return type.
   */
  Type* returnType;
};
}
