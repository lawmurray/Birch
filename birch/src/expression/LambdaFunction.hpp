/**
 * @file
 */
#pragma once

#include "src/expression/Expression.hpp"
#include "src/common/Parameterised.hpp"
#include "src/common/ReturnTyped.hpp"
#include "src/common/Braced.hpp"

namespace birch {
/**
 * Lambda function.
 *
 * @ingroup expression
 */
class LambdaFunction: public Expression,
    public Parameterised,
    public ReturnTyped,
    public Braced {
public:
  /**
   * Constructor.
   *
   * @param params Parameters.
   * @param returnType Return type.
   * @param braces Braces expression.
   * @param loc Location.
   */
  LambdaFunction(Expression* params, Type* returnType, Statement* braces,
      Location* loc = nullptr);

  virtual void accept(Visitor* visitor) const;
};
}
