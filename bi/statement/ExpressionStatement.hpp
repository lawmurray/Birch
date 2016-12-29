/**
 * @file
 */
#pragma once

#include "bi/statement/Statement.hpp"
#include "bi/expression/Expression.hpp"

namespace bi {
/**
 * ExpressionStatement.
 *
 * @ingroup compiler_statement
 */
class ExpressionStatement: public Statement {
public:
  /**
   * Constructor.
   *
   * @param expr Expression.
   * @param loc Location.
   */
  ExpressionStatement(Expression* expr, shared_ptr<Location> loc = nullptr);

  /**
   * Destructor.
   */
  virtual ~ExpressionStatement();

  virtual Statement* acceptClone(Cloner* visitor) const;
  virtual void acceptModify(Modifier* visitor);
  virtual void accept(Visitor* visitor) const;

  virtual bool operator<=(Statement& o);
  virtual bool operator==(const Statement& o) const;

  /**
   * Expression.
   */
  unique_ptr<Expression> expr;
};
}

inline bi::ExpressionStatement::~ExpressionStatement() {
  //
}
