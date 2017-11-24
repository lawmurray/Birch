/**
 * @file
 */
#pragma once

#include "bi/type/Type.hpp"
#include "bi/type/EmptyType.hpp"

namespace bi {
/**
 * ReturnTyped expression or statement.
 *
 * @ingroup birch_common
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
