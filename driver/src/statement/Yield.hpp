/**
 * @file
 */
#pragma once

#include "src/statement/Statement.hpp"
#include "src/common/Numbered.hpp"
#include "src/common/Single.hpp"
#include "src/expression/Expression.hpp"
#include "src/expression/NamedExpression.hpp"

namespace birch {
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
};
}
