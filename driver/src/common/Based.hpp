/**
 * @file
 */
#pragma once

#include "src/type/Type.hpp"

namespace birch {
/**
 * Type with base.
 *
 * @ingroup common
 */
class Based {
public:
  /**
   * Constructor.
   *
   * @param base Base type.
   * @param alias Is this an alias relationship?
   */
  Based(Type* base, const bool alias);

  /**
   * Is this class an alias for another?
   */
  bool isAlias() const;

  /**
   * Base type.
   */
  Type* base;

  /**
   * Is this an alias relationship?
   */
  bool alias;
};
}
