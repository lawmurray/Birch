/**
 * @file
 */
#pragma once

#include "bi/expression/Expression.hpp"
#include "bi/statement/UnaryOperator.hpp"
#include "bi/expression/Parameter.hpp"
#include "bi/common/Named.hpp"
#include "bi/common/Unary.hpp"
#include "bi/common/Reference.hpp"

namespace bi {
/**
 * Reference to binary operator.
 *
 * @ingroup compiler_expression
 */
class UnaryReference: public Expression,
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
  UnaryReference(shared_ptr<Name> name, Expression* single,
      shared_ptr<Location> loc = nullptr, const UnaryOperator* target =
          nullptr);

  /**
   * Destructor.
   */
  virtual ~UnaryReference();

  virtual Expression* accept(Cloner* visitor) const;
  virtual Expression* accept(Modifier* visitor);
  virtual void accept(Visitor* visitor) const;

  using Expression::definitely;
  using Expression::possibly;

  virtual bool dispatchDefinitely(const Expression& o) const;
  virtual bool definitely(const UnaryReference& o) const;
  virtual bool definitely(const UnaryOperator& o) const;
  virtual bool definitely(const Parameter& o) const;

  virtual bool dispatchPossibly(const Expression& o) const;
  virtual bool possibly(const UnaryReference& o) const;
  virtual bool possibly(const UnaryOperator& o) const;
  virtual bool possibly(const Parameter& o) const;
};
}
