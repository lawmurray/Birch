/**
 * @file
 */
#pragma once

#include "src/type/Type.hpp"

namespace birch {
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
   * Generic type arguments.
   */
  Type* typeArgs;
};
}
