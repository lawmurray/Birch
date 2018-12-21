/**
 * @file
 */
#pragma once

#include "bi/expression/Expression.hpp"

namespace bi {
/**
 * Wrapper around an expression to be explicitly interpreted in the local
 * scope.
 *
 * @ingroup expression
 */
class Local: public Expression {
public:
  /**
   * Constructor.
   *
   * @param loc Location.
   */
  Local(Location* loc = nullptr);

  /**
   * Destructor.
   */
  virtual ~Local();

  virtual Expression* accept(Cloner* visitor) const;
  virtual Expression* accept(Modifier* visitor);
  virtual void accept(Visitor* visitor) const;
};
}
