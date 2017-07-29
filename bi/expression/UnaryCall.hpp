/**
 * @file
 */
#pragma once

#include "bi/expression/Call.hpp"
#include "bi/common/Unary.hpp"
#include "bi/statement/UnaryOperator.hpp"

namespace bi {
/**
 * Call to a unary operator.
 *
 * @ingroup compiler_expression
 */
template<>
class OverloadedCall<UnaryOperator> : public Expression,
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
  OverloadedCall(shared_ptr<Name> name, Expression* single,
      shared_ptr<Location> loc = nullptr, UnaryOperator* target =
          nullptr);

  /**
   * Destructor.
   */
  virtual ~OverloadedCall();

  virtual Expression* accept(Cloner* visitor) const;
  virtual Expression* accept(Modifier* visitor);
  virtual void accept(Visitor* visitor) const;

  using Expression::definitely;
  using Expression::possibly;

  virtual bool dispatchDefinitely(const Expression& o) const;
  virtual bool definitely(const OverloadedCall<UnaryOperator>& o) const;
  virtual bool definitely(const UnaryOperator& o) const;
  virtual bool definitely(const Parameter& o) const;

  virtual bool dispatchPossibly(const Expression& o) const;
  virtual bool possibly(const OverloadedCall<UnaryOperator>& o) const;
  virtual bool possibly(const UnaryOperator& o) const;
  virtual bool possibly(const Parameter& o) const;

  /**
   * Identifier for operator resolution.
   */
  unique_ptr<OverloadedIdentifier<UnaryOperator>> op;
};
}
