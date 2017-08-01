/**
 * @file
 */
#pragma once

#include "bi/expression/Expression.hpp"
#include "bi/common/Unary.hpp"
#include "bi/common/Parenthesised.hpp"

namespace bi {
/**
 * Call to a function object. Also used as a placeholder during parsing for
 * calls to first-class overloadable functions before resolution.
 *
 * @ingroup compiler_expression
 */
class Call: public Expression,
    public Unary<Expression>,
    public Parenthesised {
public:
  /**
   * Constructor.
   *
   * @param single Expression.
   * @param parens Parentheses.
   * @param loc Location.
   */
  Call(Expression* single, Expression* parens, shared_ptr<Location> loc =
      nullptr);

  /**
   * Destructor.
   */
  virtual ~Call();

  virtual Expression* accept(Cloner* visitor) const;
  virtual Expression* accept(Modifier* visitor);
  virtual void accept(Visitor* visitor) const;
};
}
