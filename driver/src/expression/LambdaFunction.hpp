/**
 * @file
 */
#pragma once

#include "src/expression/Expression.hpp"
#include "src/common/Parameterised.hpp"
#include "src/common/ReturnTyped.hpp"
#include "src/common/Braced.hpp"
#include "src/common/Scoped.hpp"

namespace birch {
/**
 * Lambda function.
 *
 * @ingroup expression
 */
class LambdaFunction: public Expression,
    public Parameterised,
    public ReturnTyped,
    public Scoped,
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

  /**
   * Destructor.
   */
  virtual ~LambdaFunction();

  virtual Expression* accept(Cloner* visitor) const;
  virtual Expression* accept(Modifier* visitor);
  virtual void accept(Visitor* visitor) const;
};
}
