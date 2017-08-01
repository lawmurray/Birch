/**
 * @file
 */
#pragma once

#include "bi/expression/Expression.hpp"
#include "bi/common/Single.hpp"

namespace bi {
/**
 * Index expression.
 *
 * @ingroup compiler_expression
 */
class Index: public Expression, public Single<Expression> {
public:
  /**
   * Constructor.
   *
   * @param single Expression.
   * @param loc Location.
   */
  Index(Expression* single, Location* loc = nullptr);

  /**
   * Destructor.
   */
  virtual ~Index();

  virtual Expression* accept(Cloner* visitor) const;
  virtual Expression* accept(Modifier* visitor);
  virtual void accept(Visitor* visitor) const;
};
}
