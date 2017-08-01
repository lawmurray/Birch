/**
 * @file
 */
#pragma once

#include "bi/statement/Statement.hpp"
#include "bi/expression/Expression.hpp"
#include "bi/common/Unary.hpp"

namespace bi {
/**
 * Yield statement.
 *
 * @ingroup compiler_statement
 */
class Yield: public Statement, public Unary<Expression> {
public:
  /**
   * Constructor.
   *
   * @param single Expression.
   * @param loc Location.
   */
  Yield(Expression* single, shared_ptr<Location> loc = nullptr);

  /**
   * Destructor.
   */
  virtual ~Yield();

  virtual Statement* accept(Cloner* visitor) const;
  virtual Statement* accept(Modifier* visitor);
  virtual void accept(Visitor* visitor) const;
};
}
