/**
 * @file
 */
#pragma once

#include "bi/statement/Statement.hpp"
#include "bi/common/Conditioned.hpp"
#include "bi/expression/EmptyExpression.hpp"
#include "bi/common/Braced.hpp"
#include "bi/common/Scoped.hpp"
#include "bi/primitive/unique_ptr.hpp"

namespace bi {
/**
 * Conditional.
 *
 * @ingroup compiler_statement
 */
class Conditional: public Statement,
    public Conditioned,
    public Braced,
    public Scoped {
public:
  /**
   * Constructor.
   *
   * @param cond Condition.
   * @param braces True branch.
   * @param falseBraces False branch.
   * @param loc Location.
   */
  Conditional(Expression* cond, Expression* braces, Expression* falseBraces,
      shared_ptr<Location> loc = nullptr);

  /**
   * Destructor.
   */
  virtual ~Conditional();

  virtual Statement* accept(Cloner* visitor) const;
  virtual Statement* accept(Modifier* visitor);
  virtual void accept(Visitor* visitor) const;

  /**
   * Block if condition is false. May be empty if there is no else clause.
   */
  unique_ptr<Expression> falseBraces;

  virtual bool dispatch(Statement& o);
  virtual bool le(Conditional& o);
};
}
