/**
 * @file
 */
#pragma once

#include "bi/expression/Expression.hpp"
#include "bi/common/Unary.hpp"

namespace bi {
/**
 * Index expression.
 *
 * @ingroup compiler_expression
 */
class Index: public Expression, public Unary<Expression> {
public:
  /**
   * Constructor.
   *
   * @param single Expression.
   * @param loc Location.
   */
  Index(Expression* single, shared_ptr<Location> loc = nullptr);

  /**
   * Destructor.
   */
  virtual ~Index();

  virtual Expression* accept(Cloner* visitor) const;
  virtual Expression* accept(Modifier* visitor);
  virtual void accept(Visitor* visitor) const;
};
}
