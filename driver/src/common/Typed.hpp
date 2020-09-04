/**
 * @file
 */
#pragma once

#include "src/type/Type.hpp"
#include "src/type/EmptyType.hpp"

namespace birch {
/**
 * Typed expression or statement.
 *
 * @ingroup common
 */
class Typed {
public:
  /**
   * Constructor.
   *
   * @param type Type.
   */
  Typed(Type* type);

  /**
   * Destructor.
   */
  virtual ~Typed() = 0;

  /**
   * Type.
   */
  Type* type;
};
}
