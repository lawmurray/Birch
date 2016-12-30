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

  virtual Expression* acceptClone(Cloner* visitor) const;
  virtual void acceptModify(Modifier* visitor);
  virtual void accept(Visitor* visitor) const;

  virtual bool operator<=(Expression& o);
  virtual bool operator==(const Expression& o) const;
};
}
