/**
 * @file
 */
#pragma once

#include "bi/expression/Expression.hpp"
#include "bi/common/Binary.hpp"

namespace bi {
/**
 * Range expression.
 *
 * @ingroup compiler_expression
 */
class Range: public Expression, public ExpressionBinary {
public:
  /**
   * Constructor.
   *
   * @param left Left operand.
   * @param right Right operand.
   * @param loc Location.
   */
  Range(Expression* left, Expression* right, shared_ptr<Location> loc =
      nullptr);

  /**
   * Destructor.
   */
  virtual ~Range();

  virtual Expression* accept(Cloner* visitor) const;
  virtual Expression* accept(Modifier* visitor);
  virtual void accept(Visitor* visitor) const;

  using Expression::definitely;
  using Expression::possibly;

  virtual bool dispatchDefinitely(Expression& o);
  virtual bool definitely(Range& o);

  virtual bool dispatchPossibly(Expression& o);
  virtual bool possibly(Range& o);
};
}
