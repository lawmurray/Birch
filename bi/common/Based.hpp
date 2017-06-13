/**
 * @file
 */
#pragma once

#include "bi/type/Type.hpp"
#include "bi/expression/Expression.hpp"
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
   * @param baseParens Base type arguments.
   */
  Based(Type* base, Expression* baseParens);

  /**
   * Destructor.
   */
  virtual ~Based() = 0;

  /**
   * Base type.
   */
  unique_ptr<Type> base;

  /**
   * Base type arguments.
   */
  unique_ptr<Expression> baseParens;
};
}
