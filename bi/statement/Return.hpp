/**
 * @file
 */
#pragma once

#include "bi/statement/Statement.hpp"
#include "bi/expression/Expression.hpp"
#include "bi/common/Unary.hpp"

namespace bi {
/**
 * Return statement.
 *
 * @ingroup compiler_statement
 */
class Return: public Statement, public Unary<Expression> {
public:
  /**
   * Constructor.
   *
   * @param single Expression.
   * @param loc Location.
   */
  Return(Expression* single, shared_ptr<Location> loc = nullptr);

  /**
   * Destructor.
   */
  virtual ~Return();

  virtual Statement* accept(Cloner* visitor) const;
  virtual Statement* accept(Modifier* visitor);
  virtual void accept(Visitor* visitor) const;
};
}
