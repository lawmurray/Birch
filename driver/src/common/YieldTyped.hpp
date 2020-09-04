/**
 * @file
 */
#pragma once

#include "src/type/Type.hpp"
#include "src/type/EmptyType.hpp"

namespace birch {
/**
 * Fiber with yield type.
 *
 * @ingroup common
 */
class YieldTyped {
public:
  /**
   * Constructor.
   *
   * @param yieldType Yield type.
   */
  YieldTyped(Type* yieldType);

  /**
   * Destructor.
   */
  virtual ~YieldTyped() = 0;

  /**
   * Yield type.
   */
  Type* yieldType;
};
}
