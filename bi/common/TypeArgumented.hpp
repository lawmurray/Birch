/**
 * @file
 */
#pragma once

#include "bi/type/Type.hpp"

namespace bi {
/**
 * Object with generic type arguments.
 *
 * @ingroup common
 */
class TypeArgumented {
public:
  /**
   * Constructor.
   *
   * @param typeArgs Generic type arguments.
   */
  TypeArgumented(Type* typeArgs);

  /**
   * Destructor.
   */
  virtual ~TypeArgumented() = 0;

  /**
   * Generic type arguments.
   */
  Type* typeArgs;
};
}
