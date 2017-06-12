/**
 * @file
 */
#pragma once

#include "bi/type/Type.hpp"
#include "bi/type/EmptyType.hpp"
#include "bi/primitive/unique_ptr.hpp"

namespace bi {
/**
 * ReturnTyped expression or statement.
 *
 * @ingroup compiler_common
 */
class ReturnTyped {
public:
  /**
   * Constructor.
   *
   * @param returnType Return type.
   */
  ReturnTyped(Type* returnType = new EmptyType());

  /**
   * Destructor.
   */
  virtual ~ReturnTyped() = 0;

  /**
   * Return type.
   */
  unique_ptr<Type> returnType;
};
}
