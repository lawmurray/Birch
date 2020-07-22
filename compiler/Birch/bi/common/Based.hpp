/**
 * @file
 */
#pragma once

#include "bi/type/Type.hpp"

namespace bi {
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
   * Destructor.
   */
  virtual ~Based() = 0;

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
