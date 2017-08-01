/**
 * @file
 */
#pragma once

#include "bi/expression/Expression.hpp"
#include "bi/common/Bracketed.hpp"
#include "bi/common/Unary.hpp"
#include "bi/expression/Parameter.hpp"

namespace bi {
/**
 * Expression with proceeding square brackets.
 *
 * @ingroup compiler_expression
 */
class Slice: public Expression, public Unary<Expression>, public Bracketed {
public:
  /**
   * Constructor.
   *
   * @param single Expression.
   * @param brackets Brackets.
   * @param loc Location.
   */
  Slice(Expression* single, Expression* brackets, Location* loc =
      nullptr);

  /**
   * Destructor.
   */
  virtual ~Slice();

  virtual Expression* accept(Cloner* visitor) const;
  virtual Expression* accept(Modifier* visitor);
  virtual void accept(Visitor* visitor) const;
};
}
