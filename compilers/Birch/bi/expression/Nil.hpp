/**
 * @file
 */
#pragma once

#include "bi/expression/Expression.hpp"

namespace bi {
/**
 * Nil literal.
 *
 * @ingroup expression
 */
class Nil: public Expression {
public:
  /**
   * Constructor.
   *
   * @param loc Location.
   */
  Nil(Location* loc = nullptr);

  /**
   * Destructor.
   */
  virtual ~Nil();

  virtual Expression* accept(Cloner* visitor) const;
  virtual Expression* accept(Modifier* visitor);
  virtual void accept(Visitor* visitor) const;
};
}
