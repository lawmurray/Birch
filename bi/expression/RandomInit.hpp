/**
 * @file
 */
#pragma once

#include "bi/expression/Expression.hpp"
#include "bi/common/Binary.hpp"
#include "bi/expression/VarParameter.hpp"

namespace bi {
/**
 * Random variable expression.
 *
 * @ingroup compiler_expression
 */
class RandomInit: public Expression, public ExpressionBinary {
public:
  /**
   * Constructor.
   *
   * @param left Left operand.
   * @param right Right operand.
   * @param loc Location.
   */
  RandomInit(Expression* left, Expression* right, shared_ptr<Location> loc =
      nullptr);

  /**
   * Destructor.
   */
  virtual ~RandomInit();

  virtual Expression* accept(Cloner* visitor) const;
  virtual Expression* accept(Modifier* visitor);
  virtual void accept(Visitor* visitor) const;

  /**
   * Push method.
   */
  unique_ptr<Expression> push;

  virtual possibly dispatch(Expression& o);
  virtual possibly le(RandomInit& o);
  virtual possibly le(VarParameter& o);
};
}
