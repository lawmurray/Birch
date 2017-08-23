/**
 * @file
 */
#pragma once

#include "bi/statement/Statement.hpp"
#include "bi/common/Conditioned.hpp"
#include "bi/expression/EmptyExpression.hpp"
#include "bi/common/Braced.hpp"
#include "bi/common/Scoped.hpp"

namespace bi {
/**
 * If.
 *
 * @ingroup compiler_statement
 */
class If: public Statement,
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
  If(Expression* cond, Statement* braces, Statement* falseBraces,
      Location* loc = nullptr);

  /**
   * Destructor.
   */
  virtual ~If();

  virtual Statement* accept(Cloner* visitor) const;
  virtual Statement* accept(Modifier* visitor);
  virtual void accept(Visitor* visitor) const;

  /**
   * Block if condition is false. May be empty if there is no else clause.
   */
  Statement* falseBraces;

  /**
   * Scope for false block.
   */
  Scope* falseScope;
};
}
