/**
 * @file
 */
#pragma once

#include "bi/expression/Expression.hpp"
#include "bi/common/Single.hpp"

namespace bi {
/**
 * Get expression, for the return value of a fiber.
 *
 * @ingroup expression
 */
class GetReturn: public Expression, public Single<Expression> {
public:
  /**
   * Constructor.
   *
   * @param single Expression.
   * @param loc Location.
   */
  GetReturn(Expression* single, Location* loc = nullptr);

  /**
   * Destructor.
   */
  virtual ~GetReturn();

  virtual bool isAssignable() const;

  virtual Expression* accept(Cloner* visitor) const;
  virtual Expression* accept(Modifier* visitor);
  virtual void accept(Visitor* visitor) const;
};
}
