/**
 * @file
 */
#pragma once

#include "bi/expression/Expression.hpp"
#include "bi/expression/Parameter.hpp"

namespace bi {
/**
 * Self-reference to an object.
 *
 * @ingroup compiler_expression
 */
class This: public Expression {
public:
  /**
   * Constructor.
   *
   * @param loc Location.
   */
  This(shared_ptr<Location> loc = nullptr);

  /**
   * Destructor.
   */
  virtual ~This();

  virtual Expression* accept(Cloner* visitor) const;
  virtual Expression* accept(Modifier* visitor);
  virtual void accept(Visitor* visitor) const;
};
}
