/**
 * @file
 */
#pragma once

#include "bi/statement/Statement.hpp"
#include "bi/expression/Expression.hpp"
#include "bi/common/Unary.hpp"

namespace bi {
/**
 * ExpressionStatement.
 *
 * @ingroup compiler_statement
 */
class ExpressionStatement: public Statement, public ExpressionUnary {
public:
  /**
   * Constructor.
   *
   * @param single Expression.
   * @param loc Location.
   */
  ExpressionStatement(Expression* single, shared_ptr<Location> loc = nullptr);

  /**
   * Destructor.
   */
  virtual ~ExpressionStatement();

  virtual Statement* accept(Cloner* visitor) const;
  virtual Statement* accept(Modifier* visitor);
  virtual void accept(Visitor* visitor) const;

  virtual bool dispatch(Statement& o);
  virtual bool le(ExpressionStatement& o);
};
}
