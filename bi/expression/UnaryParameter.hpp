/**
 * @file
 */
#pragma once

#include "bi/expression/Expression.hpp"
#include "bi/common/Named.hpp"
#include "bi/common/Numbered.hpp"
#include "bi/common/Unary.hpp"
#include "bi/common/Scoped.hpp"
#include "bi/common/Braced.hpp"

namespace bi {
/**
 * Unary operator.
 *
 * @ingroup compiler_expression
 */
class UnaryParameter: public Expression,
    public Named,
    public Numbered,
    public Unary<Expression>,
    public Scoped,
    public Braced {
public:
  /**
   * Constructor.
   *
   * @param name Name.
   * @param single Operand.
   * @param type Return type.
   * @param braces Braces expression.
   * @param loc Location.
   */
  UnaryParameter(shared_ptr<Name> name, Expression* single, Type* type,
      Expression* braces, shared_ptr<Location> loc = nullptr);

  /**
   * Destructor.
   */
  virtual ~UnaryParameter();

  virtual Expression* accept(Cloner* visitor) const;
  virtual Expression* accept(Modifier* visitor);
  virtual void accept(Visitor* visitor) const;

  using Expression::definitely;
  using Expression::possibly;

  virtual bool dispatchDefinitely(const Expression& o) const;
  virtual bool definitely(const UnaryParameter& o) const;

  virtual bool dispatchPossibly(const Expression& o) const;
  virtual bool possibly(const UnaryParameter& o) const;
};
}
