/**
 * @file
 */
#pragma once

#include "bi/expression/Expression.hpp"
#include "bi/common/Parameterised.hpp"
#include "bi/common/ReturnTyped.hpp"
#include "bi/common/Braced.hpp"
#include "bi/common/Scoped.hpp"

namespace bi {
/**
 * Lambda function.
 *
 * @ingroup compiler_expression
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
