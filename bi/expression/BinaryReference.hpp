/**
 * @file
 */
#pragma once

#include "bi/expression/Expression.hpp"
#include "bi/expression/BinaryParameter.hpp"
#include "bi/expression/VarParameter.hpp"
#include "bi/common/Named.hpp"
#include "bi/common/Binary.hpp"
#include "bi/common/Reference.hpp"

namespace bi {
/**
 * Reference to binary operator.
 *
 * @ingroup compiler_expression
 */
class BinaryReference: public Expression, public Named, public Binary<
    Expression>, public Reference<BinaryParameter> {
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
  BinaryReference(Expression* left, shared_ptr<Name> name, Expression* right,
      shared_ptr<Location> loc = nullptr, const BinaryParameter* target =
          nullptr);

  /**
   * Destructor.
   */
  virtual ~BinaryReference();

  virtual Expression* accept(Cloner* visitor) const;
  virtual Expression* accept(Modifier* visitor);
  virtual void accept(Visitor* visitor) const;

  using Expression::definitely;
  using Expression::possibly;

  virtual bool dispatchDefinitely(const Expression& o) const;
  virtual bool definitely(const BinaryReference& o) const;
  virtual bool definitely(const BinaryParameter& o) const;
  virtual bool definitely(const VarParameter& o) const;

  virtual bool dispatchPossibly(const Expression& o) const;
  virtual bool possibly(const BinaryReference& o) const;
  virtual bool possibly(const BinaryParameter& o) const;
  virtual bool possibly(const VarParameter& o) const;
};
}
