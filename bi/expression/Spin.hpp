/**
 * @file
 */
#pragma once

#include "bi/expression/Expression.hpp"
#include "bi/common/Single.hpp"

namespace bi {
/**
 * Spin expression.
 *
 * @ingroup expression
 */
class Spin: public Expression, public Single<Expression> {
public:
  /**
   * Constructor.
   *
   * @param single Expression.
   * @param loc Location.
   */
  Spin(Expression* single, Location* loc = nullptr);

  /**
   * Destructor.
   */
  virtual ~Spin();

  virtual bool isAssignable() const;

  virtual Expression* accept(Cloner* visitor) const;
  virtual Expression* accept(Modifier* visitor);
  virtual void accept(Visitor* visitor) const;
};
}
