/**
 * @file
 */
#pragma once

#include "bi/expression/Expression.hpp"
#include "bi/statement/Statement.hpp"
#include "bi/statement/EmptyStatement.hpp"
#include "bi/primitive/unique_ptr.hpp"

namespace bi {
/**
 * Braces.
 *
 * @ingroup compiler_expression
 */
class BracesExpression: public Expression {
public:
  /**
   * Constructor.
   *
   * @param stmt Statement in braces.
   * @param loc Location.
   */
  BracesExpression(Statement* stmt, shared_ptr<Location> loc = nullptr);

  /**
   * Destructor.
   */
  virtual ~BracesExpression();

  virtual Expression* acceptClone(Cloner* visitor) const;
  virtual void acceptModify(Modifier* visitor);
  virtual void accept(Visitor* visitor) const;

  virtual bool operator<=(Expression& o);
  virtual bool operator==(const Expression& o) const;

  /**
   * Statement inside braces.
   */
  unique_ptr<Statement> stmt;
};
}

inline bi::BracesExpression::~BracesExpression() {
  //
}
