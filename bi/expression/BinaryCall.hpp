/**
 * @file
 */
#pragma once

#include "bi/expression/Call.hpp"
#include "bi/common/Binary.hpp"
#include "bi/statement/BinaryOperator.hpp"

namespace bi {
/**
 * Call to a binary operator.
 *
 * @ingroup compiler_expression
 */
template<>
class OverloadedCall<BinaryOperator>: public Expression,
    public Named,
    public Binary<Expression>,
    public Reference<BinaryOperator> {
public:
  /**
   * Constructor.
   *
   * @param left Left operand.
   * @param name Name.
   * @param right Right operand.
   * @param loc Location.
   * @param target Target.
   */
  OverloadedCall<BinaryOperator>(Expression* left, shared_ptr<Name> name, Expression* right,
      shared_ptr<Location> loc = nullptr, BinaryOperator* target =
          nullptr);

  /**
   * Destructor.
   */
  virtual ~OverloadedCall<BinaryOperator>();

  virtual Expression* accept(Cloner* visitor) const;
  virtual Expression* accept(Modifier* visitor);
  virtual void accept(Visitor* visitor) const;

  using Expression::definitely;
  using Expression::possibly;

  virtual bool dispatchDefinitely(const Expression& o) const;
  virtual bool definitely(const OverloadedCall<BinaryOperator>& o) const;
  virtual bool definitely(const BinaryOperator& o) const;
  virtual bool definitely(const Parameter& o) const;

  virtual bool dispatchPossibly(const Expression& o) const;
  virtual bool possibly(const OverloadedCall<BinaryOperator>& o) const;
  virtual bool possibly(const BinaryOperator& o) const;
  virtual bool possibly(const Parameter& o) const;

  /**
   * Identifier for operator resolution.
   */
  unique_ptr<OverloadedIdentifier<BinaryOperator>> op;
};
}
