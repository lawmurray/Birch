/**
 * @file
 */
#pragma once

#include "bi/expression/Expression.hpp"
#include "bi/common/Unary.hpp"
#include "bi/common/Reference.hpp"
#include "bi/expression/Parameter.hpp"

namespace bi {
/**
 * Call to a unary operator.
 *
 * @ingroup compiler_expression
 */
class UnaryCall: public Expression,
    public Named,
    public Unary<Expression>,
    public Reference<UnaryOperator> {
public:
  /**
   * Constructor.
   *
   * @param name Name.
   * @param single Operand.
   * @param loc Location.
   * @param target Target.
   */
  UnaryCall(shared_ptr<Name> name, Expression* single,
      shared_ptr<Location> loc = nullptr, const UnaryOperator* target =
          nullptr);

  /**
   * Destructor.
   */
  virtual ~UnaryCall();

  virtual Expression* accept(Cloner* visitor) const;
  virtual Expression* accept(Modifier* visitor);
  virtual void accept(Visitor* visitor) const;

  using Expression::definitely;
  using Expression::possibly;

  virtual bool dispatchDefinitely(const Expression& o) const;
  virtual bool definitely(const UnaryCall& o) const;
  virtual bool definitely(const UnaryOperator& o) const;
  virtual bool definitely(const Parameter& o) const;

  virtual bool dispatchPossibly(const Expression& o) const;
  virtual bool possibly(const UnaryCall& o) const;
  virtual bool possibly(const UnaryOperator& o) const;
  virtual bool possibly(const Parameter& o) const;
};
}
