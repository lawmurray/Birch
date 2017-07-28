/**
 * @file
 */
#pragma once

#include "bi/expression/Expression.hpp"
#include "bi/common/Binary.hpp"
#include "bi/common/Reference.hpp"
#include "bi/expression/Parameter.hpp"

namespace bi {
/**
 * Call to a binary operator.
 *
 * @ingroup compiler_expression
 */
class BinaryCall: public Expression,
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
  BinaryCall(Expression* left, shared_ptr<Name> name, Expression* right,
      shared_ptr<Location> loc = nullptr, const BinaryOperator* target =
          nullptr);

  /**
   * Destructor.
   */
  virtual ~BinaryCall();

  virtual Expression* accept(Cloner* visitor) const;
  virtual Expression* accept(Modifier* visitor);
  virtual void accept(Visitor* visitor) const;

  using Expression::definitely;
  using Expression::possibly;

  virtual bool dispatchDefinitely(const Expression& o) const;
  virtual bool definitely(const BinaryCall& o) const;
  virtual bool definitely(const BinaryOperator& o) const;
  virtual bool definitely(const Parameter& o) const;

  virtual bool dispatchPossibly(const Expression& o) const;
  virtual bool possibly(const BinaryCall& o) const;
  virtual bool possibly(const BinaryOperator& o) const;
  virtual bool possibly(const Parameter& o) const;
};
}
