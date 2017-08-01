/**
 * @file
 */
#pragma once

#include "bi/statement/Statement.hpp"
#include "bi/expression/Expression.hpp"
#include "bi/common/Single.hpp"

namespace bi {
/**
 * ExpressionStatement.
 *
 * @ingroup compiler_statement
 */
class ExpressionStatement: public Statement, public Single<Expression> {
public:
  /**
   * Constructor.
   *
   * @param single Expression.
   * @param loc Location.
   */
  ExpressionStatement(Expression* single, Location* loc = nullptr);

  /**
   * Destructor.
   */
  virtual ~ExpressionStatement();

  virtual Statement* accept(Cloner* visitor) const;
  virtual Statement* accept(Modifier* visitor);
  virtual void accept(Visitor* visitor) const;
};
}
