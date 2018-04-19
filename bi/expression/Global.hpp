/**
 * @file
 */
#pragma once

#include "bi/expression/Expression.hpp"

namespace bi {
/**
 * Keyword `global`.
 *
 * @ingroup expression
 */
class Global: public Expression {
public:
  /**
   * Constructor.
   *
   * @param loc Location.
   */
  Global(Location* loc = nullptr);

  /**
   * Destructor.
   */
  virtual ~Global();

  virtual Expression* accept(Cloner* visitor) const;
  virtual Expression* accept(Modifier* visitor);
  virtual void accept(Visitor* visitor) const;
};
}
