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

  using Expression::definitely;
  using Expression::possibly;

  virtual bool dispatchDefinitely(const Expression& o) const;
  virtual bool definitely(const This& o) const;
  virtual bool definitely(const Parameter& o) const;

  virtual bool dispatchPossibly(const Expression& o) const;
  virtual bool possibly(const This& o) const;
  virtual bool possibly(const Parameter& o) const;
};
}
