/**
 * @file
 */
#pragma once

#include "bi/expression/Expression.hpp"
#include "bi/common/Parenthesised.hpp"
#include "bi/common/Unary.hpp"
#include "bi/common/Scoped.hpp"
#include "bi/expression/VarParameter.hpp"

namespace bi {
/**
 * Lambda initialisation expression.
 *
 * @ingroup compiler_expression
 */
class LambdaInit: public Expression,
    public Parenthesised,
    public ExpressionUnary,
    public Scoped {
public:
  /**
   * Constructor.
   *
   * @param parens Parentheses.
   * @param single Expression.
   * @param loc Location.
   */
  LambdaInit(Expression* parens, Expression* single,
      shared_ptr<Location> loc = nullptr);

  /**
   * Destructor.
   */
  virtual ~LambdaInit();

  virtual Expression* accept(Cloner* visitor) const;
  virtual Expression* accept(Modifier* visitor);
  virtual void accept(Visitor* visitor) const;

  using Expression::definitely;
  using Expression::possibly;

  virtual bool dispatchDefinitely(Expression& o);
  virtual bool definitely(LambdaInit& o);
  virtual bool definitely(VarParameter& o);

  virtual bool dispatchPossibly(Expression& o);
  virtual bool possibly(LambdaInit& o);
  virtual bool possibly(VarParameter& o);
};
}
