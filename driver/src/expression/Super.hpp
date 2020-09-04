/**
 * @file
 */
#pragma once

#include "src/expression/Expression.hpp"

namespace birch {
/**
 * Keyword `super`.
 *
 * @ingroup expression
 */
class Super: public Expression {
public:
  /**
   * Constructor.
   *
   * @param loc Location.
   */
  Super(Location* loc = nullptr);

  /**
   * Destructor.
   */
  virtual ~Super();

  virtual bool isSuper() const;

  virtual Expression* accept(Cloner* visitor) const;
  virtual Expression* accept(Modifier* visitor);
  virtual void accept(Visitor* visitor) const;
};
}
