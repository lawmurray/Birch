/**
 * @file
 */
#pragma once

#include "bi/statement/Statement.hpp"
#include "bi/common/Numbered.hpp"
#include "bi/common/Single.hpp"
#include "bi/expression/Expression.hpp"
#include "bi/expression/NamedExpression.hpp"

#include <list>

namespace bi {
/**
 * Yield statement.
 *
 * @ingroup statement
 */
class Yield: public Statement, public Numbered, public Single<Expression> {
public:
  /**
   * Constructor.
   *
   * @param single Expression.
   * @param loc Location.
   */
  Yield(Expression* single, Location* loc = nullptr);

  /**
   * Destructor.
   */
  virtual ~Yield();

  virtual Statement* accept(Cloner* visitor) const;
  virtual Statement* accept(Modifier* visitor);
  virtual void accept(Visitor* visitor) const;

  /**
   * Resume function.
   */
  Statement* resume;

  /**
   * State. This is a list of expressions naming the parameters and local
   * variables that must be preserved in the state for the fiber to be
   * resumed from this point.
   */
  std::list<NamedExpression*> state;
};
}
