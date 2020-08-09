/**
 * @file
 */
#pragma once

#include "bi/type/Type.hpp"
#include "bi/type/EmptyType.hpp"

namespace bi {
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
