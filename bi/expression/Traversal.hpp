/**
 * @file
 */
#pragma once

#include "bi/expression/Expression.hpp"
#include "bi/common/Binary.hpp"
#include "bi/expression/VarParameter.hpp"

namespace bi {
/**
 * Traversal expression.
 *
 * @ingroup compiler_expression
 */
class Traversal: public Expression, public ExpressionBinary {
public:
  /**
   * Constructor.
   *
   * @param left Left operand.
   * @param right Right operand.
   * @param loc Location.
   */
  Traversal(Expression* left, Expression* right, shared_ptr<Location> loc =
      nullptr);

  /**
   * Destructor.
   */
  virtual ~Traversal();

  virtual Expression* accept(Cloner* visitor) const;
  virtual Expression* accept(Modifier* visitor);
  virtual void accept(Visitor* visitor) const;

  virtual bool dispatch(Expression& o);
  virtual bool le(Traversal& o);
  virtual bool le(VarParameter& o);
};
}
