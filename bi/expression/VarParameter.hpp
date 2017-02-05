/**
 * @file
 */
#pragma once

#include "bi/expression/Expression.hpp"
#include "bi/expression/EmptyExpression.hpp"
#include "bi/common/Named.hpp"
#include "bi/common/Parenthesised.hpp"
#include "bi/common/Parameter.hpp"
#include "bi/primitive/unique_ptr.hpp"

namespace bi {
/**
 * Variable parameter.
 *
 * @ingroup compiler_expression
 */
class VarParameter: public Expression,
    public Named,
    public Parenthesised,
    public Parameter<Expression> {
public:
  /**
   * Constructor.
   *
   * @param name Name.
   * @param type Type.
   * @param parens Constructor arguments.
   * @param value Initial value.
   * @param loc Location.
   */
  VarParameter(shared_ptr<Name> name, Type* type, Expression* parens =
      new EmptyExpression(), Expression* value = new EmptyExpression(),
      shared_ptr<Location> loc = nullptr);

  /**
   * Destructor.
   */
  virtual ~VarParameter();

  virtual Expression* accept(Cloner* visitor) const;
  virtual Expression* accept(Modifier* visitor);
  virtual void accept(Visitor* visitor) const;

  /**
   * Default/initial value.
   */
  unique_ptr<Expression> value;

  virtual bool dispatchDefinitely(Expression& o);
  virtual bool definitely(VarParameter& o);

  virtual bool dispatchPossibly(Expression& o);
  virtual bool possibly(VarParameter& o);
};
}
