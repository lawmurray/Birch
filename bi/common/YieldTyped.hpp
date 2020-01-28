/**
 * @file
 */
#pragma once

#include "bi/type/Type.hpp"
#include "bi/type/EmptyType.hpp"

namespace bi {
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
