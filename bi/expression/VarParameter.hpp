/**
 * @file
 */
#pragma once

#include "bi/expression/Expression.hpp"
#include "bi/expression/EmptyExpression.hpp"
#include "bi/common/Named.hpp"
#include "bi/common/Parenthesised.hpp"
#include "bi/primitive/unique_ptr.hpp"

namespace bi {
/**
 * Variable parameter.
 *
 * @ingroup compiler_expression
 */
class VarParameter: public Expression, public Named, public Parenthesised {
public:
  /**
   * Constructor.
   *
   * @param name Name.
   * @param type Type.
   * @param parens Constructor arguments.
   * @param value Assigned value.
   * @param member Is this a member variable?
   * @param loc Location.
   */
  VarParameter(shared_ptr<Name> name, Type* type, Expression* parens =
      new EmptyExpression(), Expression* value = new EmptyExpression(),
      const bool member = false, shared_ptr<Location> loc = nullptr);

  /**
   * Constructor. Usually used internally when constructing, e.g. the default
   * assignment operator.
   *
   * @param type Type.
   */
  VarParameter(Type* type);

  /**
   * Destructor.
   */
  virtual ~VarParameter();

  virtual Expression* accept(Cloner* visitor) const;
  virtual Expression* accept(Modifier* visitor);
  virtual void accept(Visitor* visitor) const;

  virtual bool isMember() const;

  /**
   * Default/initial value.
   */
  unique_ptr<Expression> value;

  /**
   * If this variable has a lambda type, the function associated with it.
   */
  unique_ptr<Expression> func;

  /**
   * Is this a member variable?
   */
  bool member;

  using Expression::definitely;
  using Expression::possibly;

  virtual bool dispatchDefinitely(const Expression& o) const;
  virtual bool definitely(const VarParameter& o) const;

  virtual bool dispatchPossibly(const Expression& o) const;
  virtual bool possibly(const VarParameter& o) const;
};
}
