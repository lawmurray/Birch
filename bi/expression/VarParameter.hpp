/**
 * @file
 */
#pragma once

#include "bi/expression/Expression.hpp"
#include "bi/expression/EmptyExpression.hpp"
#include "bi/common/Named.hpp"
#include "bi/common/Parenthesised.hpp"
#include "bi/common/VariableMode.hpp"
#include "bi/primitive/unique_ptr.hpp"

namespace bi {
/**
 * Variable parameter.
 *
 * @ingroup compiler_expression
 */
class VarParameter: public Expression,
    public Named,
    public VariableMode,
    public Parenthesised {
public:
  /**
   * Constructor.
   *
   * @param name Name.
   * @param type Type.
   * @param form Form.
   * @param parens Constructor arguments.
   * @param value Assigned value.
   * @param loc Location.
   */
  VarParameter(shared_ptr<Name> name, Type* type, const VariableForm form,
      Expression* parens = new EmptyExpression(), Expression* value =
          new EmptyExpression(), shared_ptr<Location> loc = nullptr);

  /**
   * Constructor. Usually used internally when constructing, e.g. the default
   * assignment operator.
   *
   * @param type Type.
   * @param form Form.
   */
  VarParameter(Type* type, const VariableForm form);

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

  virtual bool dispatchDefinitely(const Expression& o) const;
  virtual bool definitely(const VarParameter& o) const;

  virtual bool dispatchPossibly(const Expression& o) const;
  virtual bool possibly(const VarParameter& o) const;
};
}
