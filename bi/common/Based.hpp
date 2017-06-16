/**
 * @file
 */
#pragma once

#include "bi/type/Type.hpp"
#include "bi/primitive/unique_ptr.hpp"

namespace bi {
/**
 * Type with base.
 *
 * @ingroup compiler_common
 */
class Based {
public:
  /**
   * Constructor.
   *
   * @param base Base type.
   */
  Based(Type* base);

  /**
   * Destructor.
   */
  virtual ~Based() = 0;

  /**
   * Base type.
   */
  unique_ptr<Type> base;
};
}
