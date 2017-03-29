/**
 * @file
 */
#pragma once

#include "bi/expression/Expression.hpp"
#include "bi/expression/EmptyExpression.hpp"
#include "bi/common/Named.hpp"
#include "bi/common/Parameter.hpp"
#include "bi/primitive/unique_ptr.hpp"

namespace bi {
/**
 * Variable parameter.
 *
 * @ingroup compiler_expression
 */
class VarParameter: public Expression, public Named, public Parameter<
    Expression> {
public:
  /**
   * Constructor.
   *
   * @param name Name.
   * @param type Type.
   * @param value Initial value.
   * @param loc Location.
   */
  VarParameter(shared_ptr<Name> name, Type* type, Expression* value =
      new EmptyExpression(), shared_ptr<Location> loc = nullptr);

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

  /**
   * If this variable has a lambda type, the function associated with it.
   */
  unique_ptr<Expression> func;

  using Expression::definitely;
  using Expression::possibly;

  virtual bool dispatchDefinitely(Expression& o);
  virtual bool definitely(VarParameter& o);

  virtual bool dispatchPossibly(Expression& o);
  virtual bool possibly(VarParameter& o);
};
}
